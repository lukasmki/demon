#!/bin/bash

SEED=1337
DIFF=1
NATOMS=20
VOLUME=8000
TEMPERATURE=2000
MAX_STEPS=5000
SYSTEM="JSON"
OUTDIR="data/test-all/$SYSTEM-d$DIFF/"

MODELS=(
  "gemma-4-31b:gemma-4-31b"
  "gemma-4-26b-a3b:gemma-4-26b-a4b"
  "gpt-oss-20b:gpt-oss-20b"
  "gpt-oss-120b:gpt-oss-120b"
  "qwen36-27b:qwen3.6-27b"
  "qwen35-122b-a10b:qwen3.5-122b-a10b"
  "mistral-medium-35-128b:mistral-medium-3.5-128b"
)

for entry in "${MODELS[@]}"; do
  name="${entry%%:*}"
  model="${entry#*:}"
  python run.py --output "$OUTDIR" --name "$name" --model "$model" --diff "$DIFF" \
    --natoms "$NATOMS" --volume "$VOLUME" --temp "$TEMPERATURE" --max-steps "$MAX_STEPS" \
    --seed "$SEED" --system "$SYSTEM"
done
