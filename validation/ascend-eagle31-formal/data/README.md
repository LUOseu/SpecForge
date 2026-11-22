# Qwen3-30B-A3B target-generated training rows

The teacher is `Qwen/Qwen3-30B-A3B-Instruct-2507` at `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`. The public input corpus is the 464,322,921-byte `ShareGPT_V4.3_unfiltered_cleaned_split.json` file at revision `8b0048ad6ae8c22f46a78c15559dec98feef5539` of `Aeala/ShareGPT_Vicuna_unfiltered`. `prepare_prompts.py` verifies the complete source and tokenizer file digests, reuses SpecForge's `process_sharegpt_row`, and renders only the first user turn through the target's native chat template with `add_generation_prompt=True`. The target receives no source assistant message. Preparation used Python 3.12.13, Transformers 5.19.0 and Tokenizers 0.23.3; the UltraChat extension used PyArrow 26.0.1. [Source corpus](https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered/tree/8b0048ad6ae8c22f46a78c15559dec98feef5539) · [Target](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/tree/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe)

Only base conversation rows with IDs ending in `_0` are eligible. We compare first user messages with NFKC normalization, Unicode case folding and whitespace folding. A second fingerprint also folds punctuation and underscores. The second fingerprint assigns whole questions to the bootstrap, rollout training, or rollout held-out partition with bucket ranges `0–29`, `30–94`, and `95–99`. Previous smoke families and questions were removed. All 1,435 fingerprints in the frozen evaluation denylist are excluded. This protects against the tested exact and punctuation-normalized matches; paraphrases can still occur in public conversation corpora. `split-audit.json` gives pairwise counts and the denylist checksum.

The frozen teacher reservoirs contain 12,800 training candidates and 256 held-out candidates. Each eligible conversation gets a stable SHA-256 rank using seed `20261121`. Selection interleaves short prompts (12–256 input tokens), medium prompts (257–576), and long prompts (577–896) in proportions 65/25/10 until a stratum is exhausted. The first 512 reserved training prompts include 333 short, 128 medium and 51 long. Across the first 8,192 candidates the counts are 6,515, 1,266 and 411. A separate digest chooses a 384- or 640-token generation allowance (about 75/25). Sequences therefore contain at most 1,536 tokens. Natural EOS is enabled. Answers shorter than eight tokens are recorded and skipped in published training shards.

`regenerate_ids.py` sends `input_ids` directly to the local target-only SGLang `/generate` endpoint. It preserves actual `output_ids`, validates the server's prompt and completion token counts and finish reason, and emits `input_ids = prompt_ids + output_ids` with a zero/one loss mask split at that exact boundary. There is no text decoding and retokenization. The source and prompt selections are deterministic and generation uses temperature zero. Request batching and hardware arithmetic can still affect a fresh greedy decode. The recorded IDs and checksums fix the exact training dataset for this run.

## Completed run

The two target replicas generated 12,402,067 tokens in 33,052 actual requests. After filtering very short answers, the retained 32,768 training rows contain 17,105,965 sequence tokens and 12,350,505 answer-mask positions. Of these rows, 18,746 use ShareGPT prompts and 14,022 use UltraChat prompts. The teacher ended 9,530 outputs naturally and reached the configured token limit on 23,238. The 128 independently held-out rows contain 51,027 answer-mask positions. The final worker exited with zero outstanding HTTP futures. See `formal-32768.complete.json`, `evidence/train-first32768.check.json`, and `evidence/formal-generation-summary.json`.

The formal 65,536-example curriculum uses the first 24,576 verified teacher answers, the disjoint 8,192 ShareGPT original-answer rows and 32,768 UltraChat original-answer rows. One unique pass over those three files contains 31,403,369 sequence tokens and 22,731,132 masked assistant positions. `formal-corpus-65536.json` gives the three file hashes and independent counts. The final 8,192 target responses extend the teacher corpus beyond this curriculum. Training logs measure actual epoch repetitions and optimizer updates.

## Reproduction

```bash
python prepare_prompts.py --source "$SHAREGPT_JSON" --deny "$FROZEN_DENYLIST" \
  --smoke "$EARLIER_SMOKE_MANIFEST" --model "$TARGET_MODEL" --repo "$SPECFORGE_SRC" --out "$DATA/prompts"
python audit_splits.py --original "$BOOTSTRAP/families.jsonl" --teacher "$DATA/prompts" \
  --deny "$FROZEN_DENYLIST" > "$DATA/split-audit.json"
bash launch-regeneration.sh "$DATA"
```

The client runs on the same machine as both local SGLang replicas. It checks `/health`, `/server_info`, target and tokenizer config digests, BF16 precision and a minimum context length of 1,536. A process lock protects its SQLite journal. The client starts filling a healthy replica immediately and joins the second independently. Each has at most 24 outstanding requests. Per-endpoint and aggregate token counts, sampled throughput and progress are written to `regeneration-progress.json` and `regeneration.log`. Rerun the launcher after a failed process to reuse committed rows.

Immutable stage data and manifests use `train-first0016`, `train-first0128`, `train-first0512`, `train-first1024`, `train-first2048`, `train-first4096`, `train-first8192`, and `heldout-first0128`. Each is published only once all preceding candidate indexes have a durable result. A `.manifest.json` records counts, finish reasons, sequence lengths, source identity and individual file hashes. `.jsonl` is directly consumable as SpecForge `data.prompts_path`. `.audit.jsonl` records prompt and answer lengths and fingerprints. `*-sliceNNNN-NNNN.jsonl` contains only the new rows since the preceding stage. Training and held-out IDs, reversible token rows and generated completions stay in the private campaign directory. The broad map frozen from the disjoint 8,192 original-answer pairs is reused across training stages; held-out answers do not contribute to that map.

At the 512-row training manifest the second replica drains automatically. `drained-30101.json` is the publication barrier for handing off that pair of NPUs. The first replica continues generating. `audit_release.py --root "$DATA" --stage train-first0512 --vocab "$BOOTSTRAP/vocab.pt"` reopens the serialized rows, recovers prompt IDs from the frozen source selection and answer IDs from the target journal, checks exact mask boundaries and sha256s and measures the answer-token coverage of the frozen mapping.

## Formal corpus extension

The complete eligible ShareGPT teacher half contains 18,898 training and 1,412 held-out candidates. The 12,800 training rows originally frozen and the complete generated stages retain their byte-level definitions. The extension appends 6,098 further ShareGPT training prompts, followed by eligible first-user prompts from the public `HuggingFaceH4/ultrachat_200k` `train_sft` split at revision `8049631c405ae6576f93f445c6b8166f76f5505a`. All three parquet files were checked against their HF LFS SHA-256 values. Of 207,865 source rows, 100,553 additional teacher-training prompts were eligible. They come from the upper half of the punctuation-insensitive hash buckets, use only the native target-generated answers, and are checked against every eligible ShareGPT exact and near fingerprint and the evaluation denylist. The total teacher-training reserve is **119,451**. [UltraChat source](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/tree/8049631c405ae6576f93f445c6b8166f76f5505a/data)

`capacity-manifest.json` records the immutable prefix and tail SHA-256s and exact capacity. `formal-split-audit.json` independently checks all 119,451 teacher-training candidates, all 256 reserved teacher-holdout candidates, 8,192 original-answer ShareGPT examples, and 32,768 original-answer UltraChat examples. All six pairwise family, normalized-question, and punctuation-normalized-question intersections were zero. All four groups had zero exact matches to the 1,435 frozen evaluation-question hashes.

The formal generation target is 32,768 genuine target-answer rows, with checkpoints at 12,288, 16,384 and 24,576 additional to the initial 8,192 stage. The 24,576 stage plus 40,960 separate original-answer rows supports 65,536 unique training examples. `extend-regeneration.sh` waits for atomic publication of `prompts-extension-v1/manifest.json` and the current client's exit. It then reopens the same SQLite journal, verifies the original input identity and tail hash, resumes completed rows and continues to `train-first32768.manifest.json`. A failed extension process is retried after 30 seconds. `watch-scale-audits.sh` independently checks each formal data release and the 32k vocabulary coverage. This controller does not start or terminate SGLang servers.

A service owner can release the formal generation servers after `formal-32768.complete.json` is atomically published. `confirm_completion.py` requires the final producer's own completed record, zero outstanding HTTP futures, exit of that worker, 32,768 accepted training rows plus 128 held-out rows in SQLite and independently validated token files for both. The marker records the three SHA-256 values for each stage (training JSONL, manifest, audit), supervised and total token counts, and the worker's done-record hash.

The following commands build the formal append from the same original selections and public data files. `ULTRACHAT_SHARDS` contains the three `train_sft-*.parquet` files of the pinned revision. `ORIGINAL_SHAREGPT_FAMILIES` and `ORIGINAL_ULTRACHAT_FAMILIES` are the independently frozen original-label split manifests. `DATA` contains the initial `prompts` directory and the live SQLite journal.

```bash
python prepare_prompts.py --all-eligible --source "$SHAREGPT_JSON" --deny "$FROZEN_DENYLIST" \
  --smoke "$EARLIER_SMOKE_MANIFEST" --model "$TARGET_MODEL" --repo "$SPECFORGE_SRC" --out "$WORK/sharegpt-all"
python build_blocklist.py --original "$ORIGINAL_SHAREGPT_FAMILIES" --eligible "$WORK/sharegpt-all" \
  --source "$SHAREGPT_JSON" --smoke "$EARLIER_SMOKE_MANIFEST" --out "$WORK/sharegpt-blocked"
python extend_prompts.py --base "$DATA/prompts" --eligible "$WORK/sharegpt-all" \
  --ultra "$ULTRACHAT_SHARDS" --block "$WORK/sharegpt-blocked" --deny "$FROZEN_DENYLIST" \
  --model "$TARGET_MODEL" --repo "$SPECFORGE_SRC" --out "$WORK/extension"
python audit_splits.py --original "$ORIGINAL_SHAREGPT_FAMILIES" --original-ultra "$ORIGINAL_ULTRACHAT_FAMILIES" \
  --teacher "$DATA/prompts" --tail "$WORK/extension/train-tail.jsonl" --deny "$FROZEN_DENYLIST"
```

Copy the verified `train-tail.jsonl` to `DATA/prompts-extension-v1` first, then publish its `manifest.json`. Start `extend-regeneration.sh DATA` to continue generation and `watch-scale-audits.sh DATA` to validate formal releases. `confirm_completion.py --root DATA` issues the final service-release record. On the training host, consume a cumulative `data.prompts_path` only after its corresponding manifest exists. Read the SHA-256 from the manifest before combining new rows with a previous stage.
