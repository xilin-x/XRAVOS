#!/bin/bash

source .venv/bin/activate  # activate the virtual, conda or other environment if needed, e.g., `conda activate xravos`
conda activate xravos
export CUDA_VISIBLE_DEVICES=X  # specify the GPU device(s) to use, e.g., `export CUDA_VISIBLE_DEVICES=0,1` for using GPU 0 and 1

RANDOM=$$  # no need to modify, set random seed for master port generation, using the process ID to reduce the chance of port conflicts when running multiple training sessions simultaneously
nnodes=len(X)  # modify len(X) for specific number of nodes, set to the number of nodes for distributed training, if using multiple nodes, you may need to set up the master address and port for distributed training, e.g., `export MASTER_ADDR=master_node_ip` and `export MASTER_PORT=12345`
nproc_per_node=len(X)  # modify len(X) for specific number of nodes, set to the number of GPUs per node, e.g., `nproc_per_node=4` for using 4 GPUs on a single node

static_root="data/static"
yv_root="data/YouTube"
davis_root="data/DAVIS"
xray_root="data/MOSXAV"

stage=0  # set to the training stage, e.g., 0 for stage 1, 1 for stage 2, etc. You can run different stages sequentially by changing this variable and re-running the script.
size_window=13  # set to the size of the temporal window for training, e.g., 13 for using a sampling window
batch_size=16  # set to the batch size for training, e.g., 16 or 32 depending on your GPU memory
id="xravos_sw${size_window}_s${stage}"  # set to the experiment ID for saving logs and checkpoints, e.g., `xravos_sw13_s0` for stage 1 training with a window size of 13. You can change this to organize your experiments better.

# load_network="saves/xxx.pth"  # load pre-stage pretrained model for next stage training, if needed

torchrun --nnodes=${nnodes} \
    --nproc_per_node=${nproc_per_node} \
    --master_port=$((RANDOM % 1000 + 12000)) \
    train.py \
    --size_window ${size_window} \
    --id ${id} \
    --stage ${stage} \
    --batch_size ${batch_size} \
    --static_root ${static_root} \
    --yv_root ${yv_root} \
    --davis_root ${davis_root} \
    --xray_root ${xray_root}
    # --load_network ${load_network}