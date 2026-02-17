#!/usr/bin/env bash
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${GIGAMIDI_ENV_NAME:-gigamidi}"
cd "$REPO_ROOT"
if [[ -f environment.yml ]]; then
  mamba env create -f environment.yml
else
  mamba create -n "$ENV_NAME" python=3.10 -y
  echo "Then: mamba activate $ENV_NAME && pip install -r emotion_genre/requirements.txt"
fi
echo "Activate: mamba activate $ENV_NAME"
