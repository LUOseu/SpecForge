#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${EVAL_PHYSICAL_DEVICES:?Set the reserved physical pair, in TP rank order}"
stack=${EVAL_NATIVE_STACK:-/tmp/e31-stack}
target=${EVAL_TARGET:-$stack/local-model}
stage=${1:?screen, dev or final}
arm=${2:?target, pretrained, eagle3 or eagle31}
label=${3:?stable run name}
draft=${4:-}
record=${EVAL_STATE:-$here/runs}
mkdir -p "$record/maps"
map="$record/maps/$(hostname)-${EVAL_PHYSICAL_DEVICES//,/-}.txt"
env LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}" npu-smi info -m > "$map.$$.tmp"
mv "$map.$$.tmp" "$map"
logical=$(python3 - "$EVAL_PHYSICAL_DEVICES" "$map" <<'PY'
import sys
p=sys.argv[1].split(','); rows=[x.split() for x in open(sys.argv[2])]
m={r[0]:r[2] for r in rows if len(r)>=4 and r[2].isdigit() and '910b3' in ''.join(r[3:]).lower()}
if len(p)!=2 or len(set(p))!=2 or any(x not in m for x in p): raise SystemExit('invalid physical 910B3 pair')
print(','.join(m[x] for x in p))
PY
)
export ASCEND_RT_VISIBLE_DEVICES=$logical
args=(--stage "$stage" --arm "$arm" --name "$label" --target "$target" --devices "$EVAL_PHYSICAL_DEVICES"
      --device-map "$map" --cache "$stack/scale-eval-target-cache" --port "${EVAL_PORT:-30681}"
      --out "$record/$stage/$label")
case "$arm" in
  target) ;;
  pretrained) args+=(--draft "${draft:-$here/pretrained-e3}") ;;
  eagle3|eagle31) args+=(--draft "${draft:?supply the ready SpecForge serving export}" --require-ready) ;;
  *) echo "invalid serving arm: $arm" >&2; exit 2 ;;
esac
if [[ -n ${EVAL_CELLS:-} ]]; then args+=(--cells "$EVAL_CELLS"); fi
exec "$stack/native" "$stack/native-server/bin/python" -u "$here/evaluate.py" "${args[@]}"
