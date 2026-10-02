#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=08:00:00
#SBATCH --job-name=d1_backb_adapt
#SBATCH --ntasks=1
#SBATCH --output=batch_scripts/run_logs/m1_backbone/model_1.%j.out
#SBATCH --error=batch_scripts/run_logs/m1_backbone/model_1.%j.out

lscpu
nvidia-smi
module load anaconda3/2022.05
source activate refseg

# use the node’s hostname for rendezvous
export MASTER_ADDR=$(hostname)
# pick a port in [30000…39999] based on SLURM_JOB_ID to avoid collisions
export MASTER_PORT=$((30000 + SLURM_JOB_ID % 10000))

#    --distributed \

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --use_env \
  --master_port=$MASTER_PORT \
  main.py \
    --model=ReMamber_Conv_PEFT \
    --output_dir=./outputs \
    --if_amp \
    --batch_size=8 \
    --finetune=pretrain/ReMamber_Conv.pth \
    --resume=outputs/checkpoint_model:f802c80f.pth \
    --data-set=refzom \
    --lr=9e-6