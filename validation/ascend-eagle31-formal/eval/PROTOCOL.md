# Ascend 910B3 EAGLE3.1 independent evaluation

This protocol was frozen before scale-checkpoint inference. The target is `Qwen/Qwen3-30B-A3B-Instruct-2507@0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`. The independent pretrained reference is `lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex@d1ac703a537d2b8a5b748d4f5f8ca7e97efe9214`. The published checkpoint was trained on Open-PerfectBlend. Its training data and budget differ from the campaign's exports. EAGLE3.1 uses the EAGLE3 serving algorithm with its trained feature and output normalization switches. [Reference model card](https://huggingface.co/lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex/blob/d1ac703a537d2b8a5b748d4f5f8ca7e97efe9214/README.md)

## Frozen population

| Use | Source | Questions | Decoding cap and concurrency | Replicates |
|---|---|---:|---|---:|
| Warm up each cell | GSM8K main train, disjoint from all measured rows | 4 | Same as measured cell | Excluded |
| Ongoing checkpoint screen | First half of frozen GSM8K main train development set | 16 | 256 at C1 | 1 |
| Shortlist confirmation | All frozen GSM8K main train development questions | 32 | 256 at C1/C4 | 1 |
| Primary external evaluation | GSM8K main official test | 128 | 512 at C1/C4 | 2 |
| Length sensitivity | First 32 of the same frozen GSM8K test questions | 32 | 256 at C1/C4 | 1 |
| Deployment-style supplement | First turn of MT-Bench writing and extraction, 8 per category | 16 | 256 at C1/C4 | 2 |
| Separate streaming latency companion | Same 16 MT-Bench first turns | 16 | 256 at C1/C4 | 1 |

Both sources are immutable Hugging Face file snapshots. The three downloaded inputs are checked against the SHA-256 values in `frozen/protocol.json`. Each group is selected by deterministic hash rank with seed `scale-20261121/external-eval-v1`. The denial list reserves the normalized first user question for all 1,319 official GSM8K test rows, all 36 dev/warmup questions and all 80 MT-Bench first turns, 1,435 hashes in total. The normalization is NFKC, casefold, split/join whitespace, SHA-256 of UTF-8. Training receives only this list. Evaluator-only `gold.json` contains numerical answers. This overlap check applies to this campaign's corpus, not the public model's historical training sets. Frozen tokenizer and final token-ID checksums are included. The original MT-Bench carries multi-turn prompts and requires a judge for its published score. Here the 16 first turns measure instruction-serving behavior through length, latency, throughput and equality. [GSM8K snapshot](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866) · [MT-Bench prompt snapshot](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts/tree/e3a795c5e9a82ee40611c416b8a7786c73198991)

## Checkpoint selection

Run the 16-question screen for AR, the published draft, and a sequence of actual exported checkpoints, normally steps 64, 256, 512 and 1024 or their published equivalents. If the reservation is short, measure the latest eligible milestone first. The screen reports teacher verification rounds and counters for that real serving export. Within each training variant, shortlist at most two checkpoints with the highest screen pooled acceptance length; break ties by lower step. Confirm these two on the full 32-question dev set. Select the checkpoint with highest C1 pooled acceptance length, breaking ties by C4 length and then lower step. Report any other checkpoint selected by the operator and the reason. No final-test rows enter the selection rule. Publish the dev metrics of all measured candidates to make selection visible. A released pretrained reference is a deployment comparator, not a matched-budget EAGLE3 ablation. If the training campaign supplies a matched-budget EAGLE3, evaluate it separately.

## Serving and measurements

Use one reserved physical 910B3 pair for all compared arms on a host, rank order fixed. Runtime mapping comes from the live driver. Use dedicated measurement windows, target-only and each draft in a fresh server, the same target (all 16 safetensors and tokenizer hashes), pinned SGLang source, BF16, TP2, 2,048-token context, `--max-running-requests 4`, memory fraction 0.85, Ascend attention, no radix caching and no chunked prefill. All use greedy decoding, server seed 42, natural EOS. For both draft variants use steps 3, tree top-k 1 and total draft tokens 4. Every cell includes four separate warmups with its own cap/concurrency, then identical measured requests. Trial 2 uses a different predetermined question order and reverses the order of C1/C4 scheduling. A held-open HTTP session per client worker includes network, prefill and decode in the measured client wall interval. Each finished cell is atomically committed. Re-running the same command revalidates the run identity and skips complete cells. An incomplete cell is restarted with fresh warmup and measurement; elapsed downtime is never included.

Record aggregate emitted tokens divided by entire measured wall time, median and empirical p95 per-request latency, cap hits, output tokens per verified round, and strictly accepted draft tokens divided by proposals. The acceptance denominator is the scheduler's real verification count. One-token EOS replies with zero verify rounds contribute to end-to-end throughput and quality, and have no acceptance length. For GSM8K, ask for a short solution ending on its own `#### <number>` line. Parse this final line and compare exact rational numbers (decimals and grouped thousands are normalized); unparseable answers count as incorrect. Report counts, the Wilson 95% accuracy interval for each trial, and a seeded 2,000-draw paired bootstrap for each between-arm accuracy delta. This measures the frozen subset and prompt, not the official full-set score.

For every pair, match the ordered query hashes and raw token sequences. Report bit-exact request count, average common-prefix length and first-difference positions. Always show raw throughput and answer accuracy. Label a ratio as an equal-output speedup only when every measured answer's complete token sequence matches the matched AR run for that cell. Same-arm repeat stability and both throughput observations are reported separately. Two timing replicates support a descriptive range; they do not establish a high-confidence speedup. Eventual summaries retain partial coverage and identify each missing cell.

## Execution

The local launch helper reads the current model, loaded SGLang code and reserved device map. Stage the scripts and `frozen/` together, then set `EVAL_PHYSICAL_DEVICES` to an explicitly allocated pair. The examples use physical devices 4 and 6. Adapt the physical pair to the allocated devices. Training/inference must coordinate occupancy before invocation.

```bash
cd /path/to/SpecForge/validation/ascend-eagle31-formal/eval
export EVAL_PHYSICAL_DEVICES=4,6
export EVAL_NATIVE_STACK=/tmp/e31-stack
export EVAL_TARGET=/tmp/e31-stack/local-model
bash run-pair.sh screen target ar
bash run-pair.sh screen pretrained nex-pretrained
bash run-pair.sh screen eagle31 original-step256 ../train/exports/original8k-ddp4-b2-v1-step256
python3 compare.py --run runs/screen/ar --run runs/screen/nex-pretrained --run runs/screen/original-step256 --out screen-report.json
```

For dev or final use the same command with `dev` or `final` in place of `screen`. After selection, keep the stage and label constant when resuming. A reservation can run a specified subset of cells using `EVAL_CELLS=dev-n32-o256-c4-r1`. The optional `screen_checkpoints.py` polls only requested milestone names and checks the publisher's `ready.json` against actual serving weights. This poller needs the same authorized pair and native environment as the single-checkpoint command. Every run records raw token IDs privately for recomputation. Reports omit strings, questions and answers.

At the smoke-observed 30–46 C1 output tok/s, 16 questions averaging 150 tokens would take roughly 1–1.5 minutes of timed C1 generation plus startup and warmups. A complete 672-request final arm could take 30–90 minutes; actual wall time depends on natural EOS, C4 throughput, cold kernel startup and competing jobs. Preserve actual elapsed times and independent test coverage if a reservation ends.

The non-streaming timing cells remain primary. Additional SSE cells provide TTFT, TPOT and post-first-chunk decode throughput. The report also checks SSE/non-streaming token equality. See [paper comparators and metric definitions](paper-comparisons.md).
