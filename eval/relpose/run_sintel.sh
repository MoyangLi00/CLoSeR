#!/bin/bash

set -e

workdir='.'
model_names=('ttt3r') # ttt3r cut3r

ckpt_name='cut3r_512_dpt_4_64'
model_weights="${workdir}/src/${ckpt_name}.pth"


datasets=('sintel')


for model_name in "${model_names[@]}"; do
for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/s1/relpose/${data}/${model_name}"
    echo "$output_dir"
    accelerate launch --num_processes 2 --main_process_port 29550 eval/relpose/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512 \
        --model_update_type "$model_name"
done
done


