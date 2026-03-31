#!/bin/bash

source .venv/bin/activate  # activate the virtual, conda or other environment if needed, e.g., `conda activate xravos`
conda activate xravos
export CUDA_VISIBLE_DEVICES=X  # specify the GPU device(s) to use, e.g., `export CUDA_VISIBLE_DEVICES=0,1` for using GPU 0 and 1

model="saves/xxx.pth"  # specify the path to the trained model checkpoint for evaluation, e.g., `saves/xravos_sw13_s3.pth` for evaluating the stage 3 model trained with a window size of 13. Make sure to change this to the correct path of your trained model checkpoint.
s_w=13
mem_every=7
data_path="data/MOSXAV/trainval"  # trainval or test
split="val"  # val or test, val --> trainval split for evaluation, test --> test split for evaluation
output="results/mosxav_${split}"

python eval_mosxav.py \
    --s_w ${s_w} \
    --mem_every ${mem_every} \
    --model ${model} \
    --data_path ${data_path} \
    --split ${split} \
    --output ${output}
