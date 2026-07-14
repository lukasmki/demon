#!/bin/bash

SEED=1337
DIFF=1
NATOMS=20
VOLUME=8000
TEMPERATURE=2000
MAX_STEPS=5000
MODEL="gemma-4-26b-a4b"
OUTDIR="data/test-system"

python run.py --output "$OUTDIR" --name "$MODEL-JSON" \
  --model "$MODEL" --diff "$DIFF" --natoms "$NATOMS" \
  --volume "$VOLUME" --temp "$TEMPERATURE" \
  --max-steps "$MAX_STEPS" --seed "$SEED" --system "JSON"

python run.py --output "$OUTDIR" --name "$MODEL-XYZ" \
  --model "$MODEL" --diff "$DIFF" --natoms "$NATOMS" \
  --volume "$VOLUME" --temp "$TEMPERATURE" \
  --max-steps "$MAX_STEPS" --seed "$SEED" --system "XYZ"
