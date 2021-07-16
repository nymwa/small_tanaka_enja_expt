#!/bin/bash

set -euxi

fairseq-interactive data-bin \
    --buffer-size 1024 \
    --batch-size 128 \
    --path checkpoints/checkpoint50.pt \
    --beam 4 \
    --lenpen 0.6 \
    < test.en \
    | grep '^H' \
    | cut -f 3 \
    | python src/decode.py \
    | tee output.txt \
    | sacrebleu small_parallel_enja/test.ja

