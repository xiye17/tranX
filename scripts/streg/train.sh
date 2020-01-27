#!/bin/bash
set -e

seed=${1:-0}
vocab="data/streg/vocab.freq2.bin"
train_file="data/streg/train.bin"
dev_file="data/streg/val.bin"
dropout=0.4
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=32
type_embed_size=32
lr_decay=0.985
lr_decay_after_epoch=20
max_epoch=100
patience=5 
beam_size=5
batch_size=10
lr=0.0025
ls=0.1
lstm='lstm'
model_name=model.streg.sup.${lstm}.hid${hidden_size}.embed${embed_size}.act${action_embed_size}.field${field_embed_size}.type${type_embed_size}.drop${dropout}.lr_decay${lr_decay}.lr_dec_aft${lr_decay_after_epoch}.beam${beam_size}.$(basename ${vocab}).$(basename ${train_file}).pat${patience}.max_ep${max_epoch}.batch${batch_size}.lr${lr}.glorot.no_par_info.no_copy.ls${ls}.seed${seed}

echo "**** Writing results to logs/streg/${model_name}.log ****"
mkdir -p logs/streg
echo commit hash: `git rev-parse HEAD` > logs/streg/${model_name}.log

python -u exp.py \
    --cuda \
    --seed ${seed} \
    --mode train \
    --batch_size ${batch_size} \
    --asdl_file asdl/lang/streg/streg_asdl.txt \
    --transition_system streg \
    --train_file ${train_file} \
    --dev_file ${dev_file} \
    --vocab ${vocab} \
    --lstm ${lstm} \
    --primitive_token_label_smoothing ${ls} \
    --no_parent_field_type_embed \
    --no_parent_production_embed \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience ${patience} \
    --max_epoch ${max_epoch} \
    --lr ${lr} \
    --no_copy \
    --lr_decay ${lr_decay} \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --decay_lr_every_epoch \
    --glorot_init \
    --beam_size ${beam_size} \
    --decode_max_time_step 110 \
    --log_every 50 \
    --save_to saved_models/streg/${model_name} 2>&1 | tee -a logs/streg/${model_name}.log

. scripts/streg/test.sh saved_models/streg/${model_name}.bin 2>&1 | tee -a logs/streg/${model_name}.log