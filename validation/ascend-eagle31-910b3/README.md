# Qwen3 EAGLE3.1 online training on Ascend 910B3

This bundle reproduces an eight-update SpecForge integration run with `Qwen/Qwen3-30B-A3B-Instruct-2507`. SGLang ran the target on two NPUs and published the three auxiliary states and final hidden state to Mooncake. A third NPU trained the draft with BF16 and SDPA. The control ran the EAGLE3 preset on the same eight examples. The training source in this checkout matches upstream `5d4c6a1483afec88b19270d369d16b99492abfdf`.

## Environment

The measured notebook used the aarch64 filesystem of the following 910B image with the notebook's own NPU devices and driver libraries. The driver was 26.0.rc1, firmware 9.0.0.0.205, CANN/HCCL 9.1.0 and each card had 65,536 MiB HBM. The commands below start inside this image or its extracted userland. The notebook used native execution of its extracted Python and CANN libraries. A Docker-capable host can start the same image with its normal Ascend device and driver mounts. See [SGLang's Ascend setup](https://github.com/sgl-project/sglang/tree/c757c4bb7c55d1eb69fdb2bdc6e6fb8507acb7a0/docs/docs/hardware-platforms/ascend-npus/getting-started) and [the measured versions](runtime/versions.json).

```text
quay.io/ascend/sglang@sha256:82532c070427448997a338f53ed0a1a01a57b27c8f77afc12cd6360ce2eac44e
```

Keep the image's PyTorch 2.10.0+cpu, torch_npu 2.10.0.post6 and NPU kernels. Pin Transformers to 5.19.0. The capture and inference servers load the SGLang revision specified by the SpecForge serving format. Run this from the repository root in the image, with the checkout and model storage mounted read/write.

```bash
export SPECFORGE_REPO="$PWD" BUNDLE="$PWD/validation/ascend-eagle31-910b3"
export WORK="${WORK:-$PWD/eagle31-work}"
mkdir -p "$WORK"
git init -q "$WORK/sglang"
git -C "$WORK/sglang" fetch --depth 1 https://github.com/sgl-project/sglang.git c757c4bb7c55d1eb69fdb2bdc6e6fb8507acb7a0
git -C "$WORK/sglang" checkout --detach FETCH_HEAD
export PYTHONPATH="$WORK/sglang/python:${PYTHONPATH:-}"
python -m venv --system-site-packages "$WORK/venv"
source "$WORK/venv/bin/activate"
python -m pip install --no-deps --no-build-isolation -e "$SPECFORGE_REPO" "$SPECFORGE_REPO/plugins/sglang-spec-capture"
python -m pip install --no-deps mooncake-transfer-engine-non-cuda==0.3.14
python -m pip install --no-deps 'accelerate==1.15.0' 'transformers==5.19.0'
python -m pip install 'datasets==4.8.4' matplotlib
python -c 'import sglang, torch_npu; from mooncake.store import MooncakeDistributedStore; print(sglang.__file__, torch_npu.__version__); assert hasattr(MooncakeDistributedStore,"mount_segment")'
python -c 'from importlib.metadata import version; assert version("torch") == "2.10.0+cpu" and version("torch-npu") == "2.10.0.post6" and version("transformers") == "5.19.0"'
command -v mooncake_master
npu-smi info -m
```

The benchmark verifies the pinned SGLang source even when the base image has its own installed SGLang distribution. Set the two capture chip logic IDs and the single trainer chip logic ID according to `npu-smi info -m`. The measured capture used physical cards 0 and 1, and the trainer used physical card 2. These had logic IDs 0, 1 and 2.

## Target and data

```bash
export TARGET_MODEL="$WORK/Qwen3-30B-A3B-Instruct-2507"
export RUN_BASE="$WORK/runs" SMOKE_DATA="$WORK/data"
export CAPTURE_DEVICES=0,1 TRAINER_DEVICE=2
hf download Qwen/Qwen3-30B-A3B-Instruct-2507 --revision 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe --local-dir "$TARGET_MODEL"
python "$BUNDLE/train/prepare.py" --repo "$SPECFORGE_REPO" --model "$TARGET_MODEL" --output "$SMOKE_DATA"
mkdir -p "$RUN_BASE"
```

Data preparation reads a checksum-verified prefix of the public ShareGPT V4.3 snapshot at `8b0048ad6ae8c22f46a78c15559dec98feef5539`. It picks one complete user/assistant exchange per eligible conversation family, uses the upstream Qwen parser, retains exchanges of at most 512 tokens, and requires at least eight supervised tokens. The first eight families supply training data, and the next eight supply held-out user questions. The training set has 2,037 input tokens and 1,363 supervised tokens. The eight examples contain 530 distinct supervised token IDs. The upstream mapping keeps those tokens and fills the remaining 31,470 draft vocabulary entries with zero-frequency IDs. The preparation script emits the split, map, and a manifest of IDs and hashes. It can run on a CPU with network access. Transfer that directory to the trainer if the notebook cannot reach Hugging Face.

## Capture, train, and export

Run the first two commands in separate foreground shells with the environment above, or background them with individual log files as shown. Each training attempt needs a fresh run directory. From the original shell:

```bash
bash "$BUNDLE/train/run.sh" master > "$RUN_BASE/master.log" 2>&1 &
MASTER_PID=$!
bash "$BUNDLE/train/run.sh" capture > "$RUN_BASE/capture.log" 2>&1 &
CAPTURE_PID=$!
until curl -fsS --max-time 5 http://127.0.0.1:30000/health >/dev/null; do
  kill -0 "$CAPTURE_PID" && kill -0 "$MASTER_PID" || { tail "$RUN_BASE/capture.log"; break; }
  sleep 5
done
for method in eagle3.1 eagle3; do
  run="qwen30b-910b3-$method"
  started=$(date +%s)
  bash "$BUNDLE/train/run.sh" train "$method" || exit 1
  echo "$(( $(date +%s) - started ))" > "$RUN_BASE/$run/wall-time-seconds.txt"
  specforge export --converter sglang --target-model "$TARGET_MODEL" \
    --checkpoint "$RUN_BASE/$run/$run-step8" --output-dir "$RUN_BASE/$run/sglang-export" || exit 1
done
kill -TERM "$CAPTURE_PID" "$MASTER_PID"
python "$BUNDLE/train/plot_loss.py" "$RUN_BASE"
python "$BUNDLE/train/summarize.py" "$RUN_BASE" "$SMOKE_DATA"
```

The launcher inherits the existing Qwen EAGLE3.1 or EAGLE3 YAML. It selects auxiliary layers 1, 23, and 44, trainer batch size 1, accumulation 1, context 512, rollout depth 7, seed 0, learning rate 0.0001, zero warmup, and one epoch of eight updates. Each producer reported eight captures and zero failures. Each step-8 directory contains a canonical model and a rank-0 optimizer/RNG checkpoint. The recorded serving hashes identify the exports used in inference. See [training results and hashes](results/train-evidence.json), [per-step losses](results/train-loss.csv), and [the loss curves](results/train-loss.png).

## Evaluate the trained drafts

Use the same two physical devices for all three arms. The measured evaluation used physical 6 and 7, corresponding to chip logic IDs 4 and 5 on this notebook. The runner takes physical IDs and resolves them through `npu-smi info -m`. If the management utility is only available outside the environment, save its output and pass `--device-map device-map.txt`. Put the verified target on local storage to reduce startup time.

```bash
python "$BUNDLE/eval/benchmark.py" --target "$TARGET_MODEL" \
  --eagle3 "$RUN_BASE/qwen30b-910b3-eagle3/sglang-export" \
  --eagle31 "$RUN_BASE/qwen30b-910b3-eagle3.1/sglang-export" \
  --devices 6,7 --held-out "$SMOKE_DATA/held-out.jsonl" --out "$RUN_BASE/eval"
```

Each arm starts a fresh SGLang server. One warmup is excluded, then the same eight prompts are sent sequentially with greedy decoding, a maximum of 128 new tokens, and EOS enabled. Both draft servers select `EAGLE3`, three speculative steps, top-k 1, and four draft tokens including the bonus verification position. The workload uses concurrency 1 and TP2. Output throughput is the measured completion-token total divided by end-to-end timed wall seconds. Pooled acceptance length is the completion-token total divided by the number of verification rounds, including the bonus token. The target-only arm has no verification rounds. The runner also records per-request token agreement against the target baseline.

## Check normalization on the trained weights

The EAGLE3.1 serving export has three feature RMSNorm scale tensors, a final RMSNorm scale, and both `fc_norm` and `norm_output` enabled. All 18 serving tensors match their canonical values byte for byte. The control has 15 matching tensors and both flags disabled. All four EAGLE3.1 scale vectors still contain BF16 ones at update eight.

With the actual trained EAGLE3.1 weights, the target's embedding table, and two fixed tokens plus seeded three-layer input features, bypassing the feature norms changed the projected activation by a maximum of 34.875. Bypassing the final RMSNorm changed logits by 5.6875 and 5.9375 at the two depths. The normalized state fed the next EAGLE3.1 step. The control fed its residual forward. These CPU checks use synthetic input features. The training and inference measurements use the real target. See the [export comparison](results/export-parity.json) and [fixed-input results](results/fixed-input-forward.json).

```bash
E3="$RUN_BASE/qwen30b-910b3-eagle3"
E31="$RUN_BASE/qwen30b-910b3-eagle3.1"
python "$BUNDLE/norm/check_checkpoint.py" \
  --eagle3 "$E3/${E3##*/}-step8" "$E3/sglang-export" \
  --eagle31 "$E31/${E31##*/}-step8" "$E31/sglang-export"
python "$BUNDLE/norm/check_trained_forward.py" --make-shells "$SPECFORGE_REPO" --shells "$WORK/model-shells"
python "$BUNDLE/norm/check_trained_forward.py" --target "$TARGET_MODEL" --shells "$WORK/model-shells" \
  --eagle3 "$E3/${E3##*/}-step8" --eagle31 "$E31/${E31##*/}-step8"
```
