#!/usr/bin/env bash
set -eu
umask 077
root=${1:?pass the absolute shared data directory}
cd "$root"
if [[ -s regeneration.pid ]] && kill -0 "$(cat regeneration.pid)" 2>/dev/null; then
  echo 'Regeneration client already running'
  exit 0
fi
nohup /bin/bash -c 'while [[ ! -f "$1/prompts/train-prompts.jsonl" || ! -f "$1/split-audit.json" ]]; do sleep 2; done; exec /usr/bin/python3 -u "$1/regenerate_ids.py" --root "$1" --servers http://127.0.0.1:30100 http://127.0.0.1:30101 --per-server 24 --train 8192 --heldout 128' _ "$root" </dev/null >>regeneration.log 2>&1 &
echo "$!" > regeneration.pid
cat regeneration.pid
