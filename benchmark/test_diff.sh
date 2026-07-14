#!/bin/bash

SEED=1337
NATOMS=20
VOLUME=8000
TEMPERATURE=2000
MAX_STEPS=5000
SYSTEM="JSON"
MODEL="gemma-4-26b-a4b"
OUTDIR="data/test-diff"
DIFFS=(1 2 3)

for DIFF in "${DIFFS[@]}"; do
  python run.py --output "$OUTDIR" --name "$MODEL-d$DIFF" --model "$MODEL" \
    --diff "$DIFF" --natoms "$NATOMS" --volume "$VOLUME" --temp "$TEMPERATURE" \ 
    --max-steps "$MAX_STEPS" --seed "$SEED" --system "$SYSTEM"
done
