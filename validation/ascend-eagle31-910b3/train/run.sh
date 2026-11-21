#!/usr/bin/env bash
# One host. Run master and capture in separate terminals before starting training.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=${SPECFORGE_REPO:?Set SPECFORGE_REPO to the upstream checkout}
model=${TARGET_MODEL:?Set TARGET_MODEL to the pinned local target directory}
python_bin=${PYTHON_BIN:-python}
cd "$repo"
export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1 MOONCAKE_PROTOCOL=tcp
export MOONCAKE_METADATA_SERVER="http://127.0.0.1:${METADATA_PORT:-35880}/metadata"
export MOONCAKE_MASTER_SERVER_ADDR="127.0.0.1:${MASTER_RPC_PORT:-35551}"
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-600}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
capture_port=${CAPTURE_PORT:-30000}
case "${1:-}" in
  master)
    exec mooncake_master --enable_http_metadata_server=true --http_metadata_server_host=127.0.0.1 \
      --http_metadata_server_port="${METADATA_PORT:-35880}" --rpc_port="${MASTER_RPC_PORT:-35551}" \
      --metrics_port="${METRICS_PORT:-35903}" --enable_metric_reporting=false --default_kv_lease_ttl=500 ;;
  capture)
    export ASCEND_RT_VISIBLE_DEVICES=${CAPTURE_DEVICES:-0,1} SPECFORGE_SPEC_CAPTURE=1
    export MOONCAKE_GLOBAL_SEGMENT_SIZE=1073741824 SGLANG_SPEC_CAPTURE_GPU_PUT=0
    exec "$python_bin" -m sglang.launch_server --model-path "$model" --dtype bfloat16 --device npu \
      --skip-tokenizer-init --tp-size 2 --context-length 1024 --attention-backend ascend \
      --chunked-prefill-size -1 --mem-fraction-static 0.80 --disable-radix-cache \
      --max-running-requests 1 --disable-cuda-graph --speculative-algorithm EAGLE3 \
      --enable-return-aux-hidden-states --return-hidden-states-mode full \
      --return-aux-hidden-state-layer-ids 1 23 44 --host 127.0.0.1 --port "$capture_port" ;;
  plan|train) ;;
  *) echo 'Usage: bash run.sh master|capture|{plan|train} {eagle3.1|eagle3}' >&2; exit 2 ;;
esac
case "${2:-}" in eagle3.1|eagle3) method=$2 ;; *) echo 'Select eagle3.1 or eagle3' >&2; exit 2 ;; esac
export ASCEND_RT_VISIBLE_DEVICES=${TRAINER_DEVICE:-2}
data=${SMOKE_DATA:-$here/data}
base=${RUN_BASE:-$here/runs}
run_id=qwen30b-910b3-$method
out=$base/$run_id
config=examples/configs/online/disaggregated/external/qwen3-30b-a3b-$method.yaml
args=(-c "$config" "model.target_model_path=$model" "model.vocab_mapping_path=$data/vocab.pt"
      'data.train_data_path=' "data.prompts_path=$data/train.jsonl" 'data.max_length=512'
      'data.build_dataset_num_proc=1' 'data.dataloader_num_workers=0'
      'training.attention_backend=sdpa' 'training.batch_size=1' 'training.accumulation_steps=1'
      'training.tp_size=1' 'training.learning_rate=0.0001' 'training.seed=0' 'training.num_epochs=1'
      'training.objective_options.ttt_length=7' 'training.max_steps=8' 'training.save_interval=8'
      'training.log_interval=1' 'training.warmup_ratio=0' 'tracking.report_to=none'
      "run_id=$run_id" "output_dir=$out" 'deployment.trainer.nproc_per_node=1'
      "deployment.disaggregated.control_dir=$out/control" "deployment.disaggregated.consumer_state_dir=$out/consumer-state"
      "deployment.disaggregated.server_urls=[\"http://127.0.0.1:$capture_port\"]"
      "deployment.disaggregated.mooncake_metadata_server=$MOONCAKE_METADATA_SERVER"
      "deployment.disaggregated.mooncake_master_server_addr=$MOONCAKE_MASTER_SERVER_ADDR")
if [[ $1 == plan ]]; then
  exec "$python_bin" -m specforge.cli train --plan "${args[@]}"
fi
curl -fsS --max-time 15 "http://127.0.0.1:$capture_port/health" >/dev/null
mkdir -p "$base"
mkdir "$out"
"$python_bin" -m specforge.cli train "${args[@]}" 2>&1 | tee "$out/train.log"
test -s "$out/$run_id-step8/training_state.pt"
