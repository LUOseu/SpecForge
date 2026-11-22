# Qwen3.5-4B MMFlash on one A100

Train and export two image-conditioned DFlash drafts for the same target. One uses ordinary DFlash loss and one uses the MMFlash image objective from Feng et al. The example captures features to disk so regeneration, scoring, training and SGLang inference can take turns on one GPU. It uses the multimodal DFlash geometry from upstream PR #730. [Paper](https://arxiv.org/abs/2611.18482) · [Author implementation](https://github.com/fscdc/MMFlash/tree/5b3fa3fe28f5bc6d7dfe75934a156c52b865dc8c) · [PR #730](https://github.com/sgl-project/SpecForge/pull/730)

## Run the A100 smoke test

Use Linux x86-64 and one A100. An 80 GB card gives headroom for this first run. Use persistent storage for the 4B target, feature cache, two training checkpoints with optimizer state, and both exports. Run from the root of this checkout. The training environment uses the versions in `pyproject.toml`, including torch 2.13.0, transformers 5.12.1 and SGLang 0.5.18. For a rental that supports a custom image, the SGLang release image is `lmsysorg/sglang:v0.5.18` (linux/amd64 manifest `sha256:bde16a8447b19e89056b9eea06c72be6c02801dc89d528c9ea90c53368fd74bf`). The recipe uses local feature files, the HF target and stock SGLang serving.

```bash
python -m pip install -e .
bash examples/mmflash/run_a100.sh all
```

To run each stage separately, pass `preflight`, `prepare`, `score`, `parity`, `train`, and `bench` in that order. `dflash` and `mmflash` run and export an individual training arm. Training starts from fresh output directories. The parity stage checks out the pinned author source, or uses an existing clean checkout supplied in `MMFLASH_REFERENCE`.

| Stage | Inputs and outputs |
| --- | --- |
| Prepare | Downloads the target at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Streams ChartQA at `b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5`. Selects 64 distinct train charts with target-generated explanations of at least 32 tokens and 12 validation charts held out by image content. The 640-pixel resize, prompt, greedy decoding with thinking off, and 128-new-token cap are shared by both arms. JSONL and images go in `cache/mmflash/chartqa`. |
| Score | The frozen target teacher-forces the stored answer twice, with and without the image. Writes per-token KL, visible entropy and fingerprints to `cache/mmflash/scores.jsonl`. Saves the with-image decoder outputs from blocks 1, 8, 15, 22 and 29 alongside expanded token IDs, loss mask and the calibrated score in `cache/mmflash/features`. |
| Parity | The pinned reference independently processes two of the cached image rows. The gate checks identical IDs/masks, raw KL and entropy, hidden states, the quantized score and nonzero visual weights. Both loss implementations receive the same draft logits and anchors. It compares FP32/BF16 objectives and logit gradients. Results go to `outputs/mmflash-smoke/parity.json`. |
| Train and export | Starting with seed 42, train each arm on the same features with the same sampler, block size 16, 32 anchors and 64 optimizer steps. Each step uses one row. Exports contain the trained weights from step 64 in `outputs/mmflash-smoke/{dflash,mmflash}/export`. |
| Bench | Starts SGLang twice, once per export, verifies the server's draft path, warms up, then sends the same 12 image prompts in order at concurrency 1 with greedy decoding. Saves per-prompt counters, elapsed decode time and output token IDs. `outputs/mmflash-smoke/comparison.json` contains both aggregates, their difference or ratio and any mismatched greedy outputs. |

The mean acceptance length is the arithmetic mean across prompts of `completion_tokens / spec_verify_ct`, including the bonus token. The report also gives the pooled ratio. Client-observed decode tokens/s uses only tokens received **after** the first nonempty streamed block, divided by the time from that block to the final token emission. This excludes prefill and measures the local client/server stream. A difference in output IDs is recorded and causes the comparison command to exit with an error.

## Train on existing image conversations

Use one JSON object per line with a stable ID, one image path, and the target's stored answer. Image paths resolve from the working directory. For direct comparison with the author encoder, use the explicit image marker and role/content fields shown below.

```json
{"id":"chart-1","image":"/data/chart-1.png","conversations":[{"role":"user","content":"<image>\nExplain the trend."},{"role":"assistant","content":"The target model's regenerated response goes here."}]}
```

Set the target and feature paths in the [training YAML](../configs/offline/colocated/qwen3.5-4b-mmflash-a100.yaml). The scorer calibrates over all scored training responses. The default example keeps training rows that fit in 1,024 expanded tokens and have at least 32 supervised positions. Text-only rows in a mixed corpus use the same prefix-confidence objective with zero visual boost.

```bash
cfg=examples/configs/offline/colocated/qwen3.5-4b-mmflash-a100.yaml
python scripts/score_visual_kl.py --config "$cfg" --input train.jsonl --scores cache/mmflash/scores.jsonl
python -m specforge.cli train --config "$cfg"
```

For assistant token `t`, compute the forward KL from the visible to the blind next-token distribution at `t-1`. Apply the authors' empirical quantile transform and multiply by `exp(-H_visible)`. This gives the visual score `g`, transported at 1e-4 resolution. In each block, let `p_i` be the draft probability of the supervised token at position `i`. The base weight at position `k` is the product of `s + (1-s) p_i` over valid predicted slots through `k`. Multiply this by `1 + alpha * g_k * (1-p_k)`. Both weights are detached. Sum the weighted cross-entropies and divide by the number of sequences in the batch. The default smoothing is 0.5 and alpha is 1.0. The DFlash control uses its standard position-decayed CE normalization. [Author objective](https://github.com/fscdc/MMFlash/blob/5b3fa3fe28f5bc6d7dfe75934a156c52b865dc8c/specforge/algorithms/common/mmflash_model.py)

## Reproduce the CPU check

The reference gate can run on deterministic logits, or on two PNGs through a miniature random Qwen3.5 with the released 4B processor. The tiny fixture sharpens its random output head to exercise the entropy gate after quantization. The example below runs the second check, trains two steps through the ordinary CLI and exports the checkpoint. Use an environment with SpecForge and torchvision installed.

```bash
F="$PWD/cache/mmflash-tiny"
ref="$PWD/cache/mmflash-author-reference"
git clone --filter=blob:none --no-checkout https://github.com/fscdc/MMFlash.git "$ref"
git -C "$ref" sparse-checkout set specforge
git -C "$ref" checkout --detach 5b3fa3fe28f5bc6d7dfe75934a156c52b865dc8c
export SPECFORGE_DEVICE=cpu CUDA_VISIBLE_DEVICES=""
python scripts/gates/make_mmflash_cpu_fixture.py --output-dir "$F"
python scripts/score_visual_kl.py --config "$F/config.yaml" --input "$F/data.jsonl" --scores "$F/scores.jsonl" --device cpu
python scripts/gates/check_mmflash_parity.py --author-repo "$ref" --target "$F/target" \
    --draft-config "$F/draft_config.json" --input "$F/data.jsonl" --scores "$F/scores.jsonl" \
    --features-dir "$F/features" --device cpu --dtype float32 --rows 2 --require-visual-signal --report "$F/parity.json"
python -m specforge.cli train --config "$F/config.yaml"
python -m specforge.cli export --to hf --checkpoint "$F/training/tiny-mmflash-step2" \
    --draft-config "$F/draft_config.json" --output-dir "$F/export"
```

The CPU check exercised 36 supervised tokens. The maximum absolute differences in KL and entropy were each below 5e-7. Quantized scores, captured layer outputs, the FP32 and BF16 losses and the corresponding gradients agreed exactly in that run. The two trained tiny draft exports reloaded and produced finite logits with the frozen target head.

The full 4B A100 result is pending. Use the single-card run to establish parity with real weights, two loadable exports, held-out acceptance length and decode speed. A 64-step comparison is an initial signal. Repeat a favorable or borderline timing result on the same card before using it to justify a larger training run.
