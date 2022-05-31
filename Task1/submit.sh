#!/bin/bash

#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --mail-type=all
#SBATCH --mail-user=chenhang20@mails.tsinghua.edu.cn

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1

module load cuda/10.0

srun python souhu_AE.py

# usage: 
#   sbatch submit.sh local/lrs2_conf_128_128_3_adamw_1e-1_blocks16.yml
#   sbatch --qos=gpugpu submit.sh local/lrs2_conf_128_128_3_adamw_1e-1_blocks16.yml