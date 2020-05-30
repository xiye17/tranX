#!/bin/bash

model_name=$(basename $1)

python exp.py \
    --mode test \
    --load_model $1 \
    --beam_size 20 \
    --test_file data/streg/testi.bin \
    --save_decode_to decodes/streg/${model_name}.testi.better.decode \
    --decode_max_time_step 90
