#!/bin/bash

SEED=1337
DIFF=1
VOLUME=8000
MAX_STEPS=5000
SYSTEM="JSON"
MODEL="gemma-4-26b-a4b"
OUTDIR="data/test-temp-density"

TEMPERATURES=(1000 2000 3000)
NUMBERS=(20 40 80)

for TEMPERATURE in "${TEMPERATURES[@]}"; do
  for NATOMS in "${NUMBERS[@]}"; do
    python run.py --output "$OUTDIR" --name "$MODEL-t$TEMPERATURE-v$VOLUME" --model "$MODEL" \
      --diff "$DIFF" --natoms "$NATOMS" --volume "$VOLUME" --temp "$TEMPERATURE" \
      --max-steps "$MAX_STEPS" --seed "$SEED" --system "$SYSTEM"
  done
done
