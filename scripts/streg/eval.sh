#!/bin/bash

model_name=$(basename $1)

python eval.py \
    --load_model $1 \
    --beam_size 5 \
    --test_file data/regex/test.bin \
    --save_decode_to decodes/regex/${model_name}.test.decode \
    --decode_max_time_step 110
