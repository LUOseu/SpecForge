#!/usr/bin/env bash
set -euo pipefail
umask 077
root=${1:?absolute data directory}
py=${2:-/root/miniconda3/envs/ms2.3_me1.0.rc2_py310/bin/python}
exec 9>"$root/scale-audit.lock"; flock -n 9 || exit 0
mkdir -p "$root/evidence"
for n in 12288 16384 24576 32768; do
  stage=train-first$n
  while [[ ! -f "$root/$stage.manifest.json" ]]; do sleep 20; done
  file="$root/evidence/$stage.check.json"; [[ ! -f "$file" ]] || continue
  "$py" "$root/audit_release.py" --root "$root" --stage "$stage" --vocab "$root/sf-e31-scale-original8192/vocab.pt" > "$file.tmp"
  mv "$file.tmp" "$file"; cp "$root/$stage.manifest.json" "$root/evidence/"
done
