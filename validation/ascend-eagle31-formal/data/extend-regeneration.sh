#!/usr/bin/env bash
set -euo pipefail
umask 077
root=${1:?absolute data directory required}
cd "$root"
exec 9>"$root/scale-supervisor.lock"
flock -n 9 || exit 0
# A byte-for-byte append extends the same immutable input sequence and SQLite history.
# The original client owns its own lock. Let it finish its 8192 target or recover it on exit.
while [[ ! -f "$root/train-first32768.manifest.json" ]]; do
  while [[ ! -f "$root/prompts-extension-v1/manifest.json" || ! -f "$root/prompts-extension-v1/train-tail.jsonl" ]]; do sleep 10; done
  previous=$(cat "$root/regeneration.pid" 2>/dev/null || true)
  if [[ -n "$previous" && -r "/proc/$previous/cmdline" ]] && tr '\0' ' ' < "/proc/$previous/cmdline" | grep -Fq "$root/regenerate_ids.py"; then sleep 15; continue; fi
  if [[ -f "$root/regeneration-done.json" && ! -f "$root/regeneration-done-8192.json" ]]; then cp "$root/regeneration-done.json" "$root/regeneration-done-8192.json"; fi
  /usr/bin/python3 -u "$root/regenerate_ids.py" --root "$root" --extension "$root/prompts-extension-v1" --servers http://127.0.0.1:30100 http://127.0.0.1:30101 --per-server 24 --train 32768 --heldout 128 >>"$root/regeneration.log" 2>&1 &
  child=$!;echo "$child" > "$root/regeneration.pid";wait "$child" || sleep 30
done
