#!/bin/bash

set -ex

TRAIN_EN=small_parallel_enja/train.en
TRAIN_JA=small_parallel_enja/train.ja
VALID_EN=small_parallel_enja/dev.en
VALID_JA=small_parallel_enja/dev.ja
TEST_EN=small_parallel_enja/test.en
TEST_JA=small_parallel_enja/test.ja

git clone https://github.com/odashi/small_parallel_enja.git
cat $TRAIN_EN $TRAIN_JA > train.enja
python src/learn.py --input train.enja --prefix tanaka --vocab-size 4000 --character-coverage 0.9995 --threads 1

encode () {
    python src/encode.py --model tanaka.model
}

encode < $TRAIN_EN > train.en
encode < $TRAIN_JA > train.ja
encode < $VALID_EN > valid.en
encode < $VALID_JA > valid.ja
encode < $TEST_EN > test.en

fairseq-preprocess -s en -t ja \
    --trainpref train \
    --validpref valid \
    --destdir data-bin \
    --joined-dictionary

