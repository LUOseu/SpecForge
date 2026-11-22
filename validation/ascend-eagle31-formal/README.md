# Qwen3-30B-A3B-Instruct-2507 EAGLE3.1 on Ascend 910B3

SpecForge `5d4c6a1483afec88b19270d369d16b99492abfdf` trained the existing one-layer EAGLE3.1 draft with BF16 SDPA, starting from newly initialized draft weights. The target was `Qwen/Qwen3-30B-A3B-Instruct-2507@0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`. Two 64 GiB 910B3 devices served the frozen target and captured layers 1, 23 and 44 via SGLang TP2. Four 64 GiB 910B3 devices ran the trainer in DDP with batch size 2 per rank. Two TP2 replicas on a second four-device host generated the Qwen responses. The final benchmark ran on one pair on the second host. The [environment manifest](runtime/versions.json) records the image digest, driver, CANN/HCCL, PyTorch, torch_npu, Mooncake and SGLang revisions.

## Training and checkpoints

| Stage | Unique rows | Epochs | Updates | LR | Last loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| [ShareGPT](results/original8k-ddp4-b2-v1) | 8,192 | 1 | 1,024 | 1e-4 | 13.3182 |
| [Qwen answers 0–512](results/teacher0000-0512-ddp4-b2-v1) | 512 | 1 | 64 | 5e-5 | 8.2277 |
| [UltraChat](results/ultrachat32768-e2-ddp4-b2-v1) | 32,768 | 2 | 8,192 | 8e-5 | 3.6776 |
| [Qwen answers 512–16,384](results/teacher0512-16384-ddp4-b2-v1) | 15,872 | 1 | 1,984 | 5e-5 | 6.4161 |
| [Qwen answers 16,384–24,576](results/teacher16384-24576-ddp4-b2-v1) | 8,192 | 1 | 1,024 | 5e-5 | 7.1392 |
| Total | 65,536 | | 12,288 | | |

The verified run consumed 98,304 sample visits and 33,789,801 supervised token positions, counting the second epoch on UltraChat. The five stages emitted 1,672 loss records at their planned logging intervals. The linked directories contain the actual YAML settings with path placeholders, full metric CSVs, and loss curves. Each phase trains against Qwen soft targets. Stages use maximum length 1,536, rollout length 7, seed 42, fixed 32K vocabulary, cosine scheduling and clipping at 0.5. The first stage uses 3% warmup, and later stages use 2%. Each new stage loads the previous weights and starts a fresh Adam state. [Training audit](results/verified-complete.json) · [Data inventory](data/formal-corpus-65536.json) · [Split audit](data/formal-split-audit.json)

The final checkpoint is `teacher16384-24576-ddp4-b2-v1-step1024`. The SGLang export used in the benchmark has SHA-256 `1edb70ae5bbc6755a4e43b85484603af635f6bcaa33a0698bfb880abb96bc9ce`. [Export record](results/final-export.json)

## MT-Bench results

Every cell timed 16 held-out MT-Bench first turns after four warmup queries, with greedy streaming and at most 256 new tokens. All three arms used the same physical pair, target snapshot and SGLang source revision. Each arm started a fresh server. Both drafts used three speculative steps, top-k 1 and four draft tokens. Acceptance length counts the verified output tokens per verification round, including the bonus token. The figure also reports strict proposal acceptance and mean per-request TPOT and TTFT.

| Concurrency | Target-only tok/s | Public Nex EAGLE3 tok/s | Trained EAGLE3.1 tok/s | EAGLE3.1 / target-only | EAGLE3.1 acceptance length |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 47.39 | 70.88 | 65.52 | 1.38× | 1.78 |
| 4 | 133.41 | 77.89 | 174.95 | 1.31× | 1.80 |

The ratios show observed end-to-end output throughput for this single timing pass per cell. The final draft matched the target-only token sequence on 7 of 16 requests at each concurrency. Nex averaged 1,856.7 ms TTFT at concurrency 4 in this pass. Nex is a public checkpoint trained on a separate corpus with a larger budget. [Benchmark figure](results/mtbench/benchmark-table.png) · [CSV](results/mtbench/benchmark-table.csv) · [Counts and token agreement](results/mtbench/report.json) · [Nex model card](https://huggingface.co/lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex)

## Reproduction

Set up the pinned Ascend environment and install the capture plugin using the [environment instructions from the eight-update run](https://github.com/LUOseu/SpecForge/blob/ascend-910b3-eagle3-smoke-20261121/validation/ascend-eagle31-910b3/README.md#environment). The source code in the root of this branch contains the exact upstream training code used. Run the following from that root with Python 3.12, Transformers 5.19.0, Tokenizers 0.23.3 and PyArrow 26.0.1. Use a fresh directory for `WORK`.

```bash
export SPECFORGE_SRC="$PWD" BUNDLE="$PWD/validation/ascend-eagle31-formal"
export WORK=/absolute/path/to/campaign
export DATA="$WORK/data" TARGET_MODEL="$WORK/target" ULTRACHAT_SHARDS="$WORK/ultrachat"
export SHAREGPT_JSON="$WORK/ShareGPT_V4.3_unfiltered_cleaned_split.json"
export EARLIER_SMOKE_MANIFEST="$BUNDLE/data/smoke-data-manifest.json"
export FROZEN_DENYLIST="$BUNDLE/eval/frozen/denylist.sha256"
export BOOTSTRAP="$DATA/sf-e31-scale-original8192"
mkdir -p "$DATA"
hf download Qwen/Qwen3-30B-A3B-Instruct-2507 --revision 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe --local-dir "$TARGET_MODEL"
curl -fL 'https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered/resolve/8b0048ad6ae8c22f46a78c15559dec98feef5539/ShareGPT_V4.3_unfiltered_cleaned_split.json' -o "$SHAREGPT_JSON"
python "$BUNDLE/data/fetch_ultrachat.py" "$ULTRACHAT_SHARDS"
python "$BUNDLE/train/prepare_original.py" --source "$SHAREGPT_JSON" --model "$TARGET_MODEL" \
  --denylist "$FROZEN_DENYLIST" --smoke-manifest "$EARLIER_SMOKE_MANIFEST" --output "$BOOTSTRAP"
python "$BUNDLE/data/prepare_prompts.py" --source "$SHAREGPT_JSON" --deny "$FROZEN_DENYLIST" \
  --smoke "$EARLIER_SMOKE_MANIFEST" --model "$TARGET_MODEL" --repo "$SPECFORGE_SRC" --out "$DATA/prompts"
python "$BUNDLE/data/prepare_prompts.py" --all-eligible --source "$SHAREGPT_JSON" --deny "$FROZEN_DENYLIST" \
  --smoke "$EARLIER_SMOKE_MANIFEST" --model "$TARGET_MODEL" --repo "$SPECFORGE_SRC" --out "$WORK/sharegpt-all"
python "$BUNDLE/data/build_blocklist.py" --original "$BOOTSTRAP/families.jsonl" --eligible "$WORK/sharegpt-all" \
  --source "$SHAREGPT_JSON" --smoke "$EARLIER_SMOKE_MANIFEST" --out "$WORK/sharegpt-blocked"
sort -u "$WORK/sharegpt-blocked/normalized.sha256" "$WORK/sharegpt-blocked/near.sha256" > "$WORK/sharegpt-exclusion.sha256"
python "$BUNDLE/data/prepare_ultrachat.py" --source "$ULTRACHAT_SHARDS" --model "$TARGET_MODEL" \
  --vocab "$BOOTSTRAP/vocab.pt" --blocklist "$WORK/sharegpt-exclusion.sha256" --denylist "$FROZEN_DENYLIST" --out "$DATA/ultrachat-main"
python "$BUNDLE/data/extend_prompts.py" --base "$DATA/prompts" --eligible "$WORK/sharegpt-all" \
  --ultra "$ULTRACHAT_SHARDS" --block "$WORK/sharegpt-blocked" --deny "$FROZEN_DENYLIST" \
  --model "$TARGET_MODEL" --repo "$SPECFORGE_SRC" --out "$WORK/extension"
```

For regeneration, run these servers and the client on the same device host, making the paths above available there. Start an instance of the following launch in each of two terminals, setting `GEN_DEVICES` to different pairs of chip logic IDs and `GEN_PORT` to 30100 and 30102. A single endpoint also works. These endpoints serve the target only.

```bash
ASCEND_RT_VISIBLE_DEVICES="$GEN_DEVICES" python -m sglang.launch_server \
  --model-path "$TARGET_MODEL" --device npu --dtype bfloat16 --tp-size 2 \
  --attention-backend ascend --context-length 2560 --mem-fraction-static 0.87 \
  --disable-radix-cache --chunked-prefill-size -1 --max-running-requests 24 \
  --cuda-graph-max-bs-decode 24 --host 127.0.0.1 --port "$GEN_PORT"
```

```bash
python "$BUNDLE/data/regenerate_ids.py" --root "$DATA" --extension "$WORK/extension" \
  --servers http://127.0.0.1:30100 http://127.0.0.1:30102 --train 24576 --heldout 128
mkdir -p "$WORK/train/campaign"
sed -n '513,16384p' "$DATA/train-first24576.jsonl" > "$WORK/train/campaign/teacher-rows-0512-16384.jsonl"
sed -n '16385,24576p' "$DATA/train-first24576.jsonl" > "$WORK/train/campaign/teacher-rows-16384-24576.jsonl"
```

This replays the data construction and generation settings. Fresh generation can vary with scheduling and backend arithmetic even at temperature zero. The audit publishes the exact training-data digests observed in this run. [Data provenance](data/README.md)

On the six-device training host, start the Mooncake master in a separate terminal. The four capture workers use the same loopback endpoint.

```bash
mooncake_master --enable_http_metadata_server=true --http_metadata_server_host=127.0.0.1 \
  --http_metadata_server_port=35880 --rpc_port=35551 --metrics_port=35903 \
  --enable_metric_reporting=false --default_kv_lease_ttl=120000
```

Start the capture service in a second terminal.

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 SPECFORGE_SPEC_CAPTURE=1 \
MOONCAKE_LOCAL_HOSTNAME=127.0.0.1 MOONCAKE_PROTOCOL=tcp \
MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata \
MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551 \
MOONCAKE_GLOBAL_SEGMENT_SIZE=12884901888 SGLANG_SPEC_CAPTURE_GPU_PUT=0 \
python -m sglang.launch_server --model-path "$TARGET_MODEL" --device npu \
  --dtype bfloat16 --tp-size 2 --attention-backend ascend --context-length 2560 \
  --mem-fraction-static 0.87 --disable-radix-cache --chunked-prefill-size -1 \
  --max-running-requests 8 --skip-tokenizer-init --disable-cuda-graph \
  --enable-return-aux-hidden-states --return-hidden-states-mode full \
  --speculative-algorithm EAGLE3 --return-aux-hidden-state-layer-ids 1 23 44 \
  --host 127.0.0.1 --port 30000
```

In a trainer terminal, set `ASCEND_RT_VISIBLE_DEVICES` using the chip logic IDs from `npu-smi info -m`. The four training chips in this run mapped to 2,3,4,5. The fixed YAMLs use placeholders for paths.

```bash
export ASCEND_RT_VISIBLE_DEVICES=2,3,4,5 SPECFORGE_DEVICE=npu
export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1 HCCL_CONNECT_TIMEOUT=600 HCCL_EXEC_TIMEOUT=600
curl -fsS http://127.0.0.1:30000/health
mkdir -p "$WORK/configs" "$WORK/logs"
set -o pipefail
for stage in original8k-ddp4-b2-v1 teacher0000-0512-ddp4-b2-v1 ultrachat32768-e2-ddp4-b2-v1 \
             teacher0512-16384-ddp4-b2-v1 teacher16384-24576-ddp4-b2-v1; do
  sed -e "s|/path/to/campaign|$WORK|g" -e "s|/path/to/target|$TARGET_MODEL|g" \
      -e "s|/path/to/SpecForge|$SPECFORGE_SRC|g" "$BUNDLE/results/$stage/run.yaml" > "$WORK/configs/$stage.yaml"
  specforge train -c "$WORK/configs/$stage.yaml" 2>&1 | tee "$WORK/logs/$stage.log" || exit 1
done
stage=teacher16384-24576-ddp4-b2-v1
specforge export --converter sglang --target-model "$TARGET_MODEL" \
  --checkpoint "$WORK/train/runs/$stage/$stage-step1024" --output-dir "$WORK/final-export"
```

Generate the evaluation inputs from their pinned sources on a CPU. The [protocol](eval/PROTOCOL.md) describes the holdout selection and comparison. `--cells` selects the two streaming cells shown above.

```bash
python "$BUNDLE/eval/lock_data.py" --source "$WORK/eval-sources" --output "$WORK/eval-frozen" --tokenizer "$TARGET_MODEL"
python "$BUNDLE/eval/fetch_reference.py" "$WORK/nex-eagle3"
npu-smi info -m > "$WORK/device-map.txt"
cells=mtbench-first-turn-n16-o256-c1-r1-stream,mtbench-first-turn-n16-o256-c4-r1-stream
args=(--stage final --port 30685 --target "$TARGET_MODEL" --devices 4,6 --device-map "$WORK/device-map.txt" \
      --cache "$WORK/hash-cache" --frozen "$WORK/eval-frozen" --cells "$cells")
python "$BUNDLE/eval/evaluate.py" "${args[@]}" --arm target --name ar --out "$WORK/eval/ar"
python "$BUNDLE/eval/evaluate.py" "${args[@]}" --arm pretrained --name nex-eagle3 --draft "$WORK/nex-eagle3" --out "$WORK/eval/nex"
python "$BUNDLE/eval/evaluate.py" "${args[@]}" --arm eagle31 --name tuned-eagle31 --draft "$WORK/final-export" --out "$WORK/eval/eagle31"
python "$BUNDLE/eval/compare.py" --run "$WORK/eval/ar" --run "$WORK/eval/nex" \
  --run "$WORK/eval/eagle31" --out "$WORK/eval/report.json"
```
