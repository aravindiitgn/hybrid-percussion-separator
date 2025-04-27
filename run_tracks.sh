#!/bin/bash

# Loop over track indexes 0 to 9.
for track_idx in {0..9}; do
    echo "Processing track index: $track_idx"
    python3 test.py \
        --track_idx "$track_idx" \
        --subset test \
        --model_path hybrid_percussion_epoch90.pth \
        --duration 10 \
        --drum_output separated_drums.wav \
        --accompaniment_output separated_accompaniment.wav
done
