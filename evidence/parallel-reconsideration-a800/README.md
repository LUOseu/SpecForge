# SpecForge A800 training pilot

Source commit `5c5bdba06b384baf18710f72f9647e2d153467af`. The target was Qwen/Qwen2.5-0.5B-Instruct at revision `7ae557604adf67be50417f59c2c2f167def9a775`. The data was the first 2,048 `train_sft` rows of HuggingFaceH4/ultrachat_200k at revision `8049631c405ae6576f93f445c6b8166f76f5505a`.

The A800 80 GB PCIe node ran driver 580.105.08, PyTorch 2.10.0+cu128, Transformers 5.18.0, and BF16 compute. `capture_sf.py` produced canonical offline records in `/dev/shm/prc-pilot/cache/sf`. It uses post-block taps [1, 11, 20] and the target's final normalized states. For reproduction, place the model snapshot under `/dev/shm/prc-pilot/models/qwen25` and a JSONL with a `conversations` field copied from `messages` under `/dev/shm/prc-pilot/data/ultrachat_2048.jsonl`.

The command used the repository's native training recipe with the target snapshot, capture path, and captured vocabulary map set as overrides. It logged every ten steps, used SDPA, set `model.use_liger_kernel=false`, and retained the latest two native checkpoints. The schedule ran to its 10,000-step limit.

The figure marks zero-gradient logs separately. The exponential moving average takes only supervised logs (span 31), and the summary's two medians each use 50 supervised logs. The observed objective decreases on this training set.
