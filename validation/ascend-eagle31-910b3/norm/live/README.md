# Observe a single EAGLE3.1 serving request

`observe_request.py` starts a disposable server after the matched timing runs. Use the same SGLang environment, target, step-eight serving export and tensor-parallel degree as the successful benchmark. This script reads its launch command from that benchmark's manifest, chooses another port and disables graph capture so that each Python hook observes actual forward calls. It requires two available 910B3 cards mapped to logical IDs `0,1` and a fresh output directory. The `--benchmark-script` argument selects the bundled `eval/benchmark.py` for process-group cleanup.

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 python norm/live/observe_request.py \
  --benchmark-script eval/benchmark.py --manifest "$BENCHMARK_OUT/manifest.json" \
  --out "$DIAGNOSTIC_OUT" --port 30973
```

Only the disposable server's children receive the observer in `PYTHONPATH`. The observer verifies the pinned EAGLE3 Python file digest and calls the original loader and forward implementations. Each real load of a norm parameter records the SHA-256 of its source and destination tensor. This detects whether the loader consumed all four weights even if every exported BF16 scale is 1. After warmup, forward hooks capture up to two rows per call in memory and calculate hashes, maximum input/output change, finiteness and relative L2 error versus the FP32 RMSNorm formula. The final norm's formula includes the decoder residual. A model hook checks whether the serving implementation returns the same normalized tensor to the head and speculative feedback. The request produces eight tokens using greedy decoding. `summary.json` contains aggregate results. Keep the full server log private. Timings from this observer have no role in the throughput comparison.
