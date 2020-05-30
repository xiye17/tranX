#!/bin/bash

model_name=$(basename $1)

python eval.py \
    --load_model $1 \
    --beam_size 5 \
    --test_file data/streg/testi.bin \
    --save_decode_to decodes/streg/${model_name}.test.decode \
    --decode_max_time_step 80
