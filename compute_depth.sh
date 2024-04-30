#!/bin/bash

export NCCL_DEBUG=info
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL



while
  port=$(shuf -n 1 -i 49152-65535)
  netstat -atun | grep -q "$port"
do
  continue
done

echo "$port"
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gn  oded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


data_path='/net/projects/willettlab/roxie62/dataset/lsun/bedroom'

python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=${port} compute_depth.py \
--data_path ${data_path} \
--temp1 1 \
--temp2 0  \
--epochs 4000 \
--batch_size 56 \
--learning_rate 2e-4 \
--weight_decay 1e-3 \
--data_type cifar10 \
--resume
#--learning_rate 5e-4 \
