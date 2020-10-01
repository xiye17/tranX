#!/bin/bash

model_name=$(basename $1)
split=$2
python exp.py \
    --mode synth \
    --load_model $1 \
    --test_file data/streg/${split}.bin \
    --save_decode_to decodes/streg/${model_name}.${split}.100-9.eval.bin \
    --decode_max_time_step 90
