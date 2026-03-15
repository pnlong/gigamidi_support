#!/usr/bin/env bash
# Usage: ./train.sh <config.yml> [extra args...]
# Calls train.py for emotion/genre configs, train_va.py for valence_arousal configs.
# Extra args are passed through (e.g. --gpu, --resume).
set -e

CONFIG="$1"
if [[ -z "$CONFIG" ]]; then
  echo "Usage: $0 <config.yml> [extra args...]" >&2
  echo "Example: $0 pretrain_model/configs/emotion_musetok.yml --gpu" >&2
  exit 1
fi
shift || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_ABS="$CONFIG"
[[ "$CONFIG" != /* ]] && CONFIG_ABS="$SCRIPT_DIR/$CONFIG"
if [[ ! -f "$CONFIG_ABS" ]]; then
  echo "Config not found: $CONFIG_ABS" >&2
  exit 1
fi

TASK=$(python3 -c "
import yaml, sys
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)
task = c.get('task')
if task in ('emotion', 'genre'):
    print('train')
else:
    print('train_va')
" "$CONFIG_ABS")

cd "$SCRIPT_DIR"
if [[ "$TASK" == "train" ]]; then
  exec python3 pretrain_model/train.py --config "$CONFIG_ABS" "$@"
else
  exec python3 pretrain_model/train_va.py --config "$CONFIG_ABS" "$@"
fi
