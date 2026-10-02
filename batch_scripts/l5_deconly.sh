#!/bin/bash
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40:2
#SBATCH --time=01:00:00
#SBATCH --job-name=l5_deconly
#SBATCH --ntasks=1
#SBATCH --output=batch_scripts/run_logs/l5_deconly/model_5.%j.out
#SBATCH --error=batch_scripts/run_logs/l5_deconly/model_5.%j.out

lscpu
nvidia-smi
source activate refseg

# use the node’s hostname for rendezvous
export MASTER_ADDR=$(hostname)
# pick a port in [30000…39999] based on SLURM_JOB_ID to avoid collisions
export MASTER_PORT=$((30000 + SLURM_JOB_ID % 10000))

#    --distributed \

python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --use_env \
  --master_port=$MASTER_PORT \
  main.py \
    --model=ReMamber_Conv \
    --output_dir=./l_outputs \
    --if_amp \
    --batch_size=14 \
    --finetune=pretrain/ReMamber_Conv.pth \
    --data-set=refzom \
    --dec-only