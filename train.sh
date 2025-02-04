#!/bin/bash

#SBATCH --account=hai_aqqua
##SBATCH --partition=develbooster
#SBATCH --partition=booster
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
# Use only physical cores. (Can use up to 2 threads per core.)
#SBATCH --threads-per-core=2
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:x00
#SBATCH --error /p/project1/hai_aqqua/hirsch2/dinov2_output/log_%j.err
#SBATCH --output /p/project1/hai_aqqua/hirsch2/dinov2_output/log_%j.out

curr_file="$(scontrol show job "$SLURM_JOB_ID" | grep '^[[:space:]]*Command=' | head -n 1 | cut -d '=' -f 2-)"
curr_dir="$(dirname "$curr_file")"

echo $curr_file $curr_dir

# Propagate the specified number of CPUs per task to each `srun`.
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

micromamba activate aqqua
# source "$curr_dir"/activate.sh

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi
export MASTER_PORT=${SLURM_JOB_ID:3:5}

echo $MASTER_ADDR $MASTER_PORT
# Prevent NCCL not figuring out how to initialize.
export NCCL_SOCKET_IFNAME=ib0
# Prevent Gloo not being able to communicate.
export GLOO_SOCKET_IFNAME=ib0

# For the example, we get a bunch of information from the
# checkpointing function about tensors being deduplicated. With this
# line, we turn off logging for those warnings. Please do not keep
# this for your scripts if you do not encounter the same problem.
# For more information, see:
# https://github.com/pytorch/pytorch/issues/117392
# export TORCH_LOGS='-torch.distributed.checkpoint._dedup_tensors'

export WANDB_API_KEY=26128e267555030ad693d8bc2196744166b08dac
export WANDB_MODE=offline
# export NCCL_DEBUG=INFO

if [ $SLURM_CLUSTER_NAME = juwels ]
then
    N_GPUS=4
else
    N_GPUS=8
fi
N_GPUS_TOTAL=$((N_GPUS*SLURM_NNODES))
N_CPUS_MAX=$(echo $SLURM_CPUS_PER_TASK | cut -d '(' -f 1)
N_CPUS=$((N_CPUS_MAX/N_GPUS))
echo "N_GPUS" $N_GPUS $N_GPUS_TOTAL "N_CPUS" $N_CPUS "JOB_CPUS_PER_NODE" $SLURM_JOB_CPUS_PER_NODE $SLURM_CPUS_PER_TASK

env | grep SLURM_ | sort
env | grep SRUN_ | sort
# exit

BATCH_S=64
srun python -m torchrun_jsc \
     --nproc_per_node=gpu \
     --nnodes="$SLURM_JOB_NUM_NODES" \
     --rdzv_id="$SLURM_JOB_ID" \
     --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
     --rdzv_backend=c10d \
     /p/home/jusers/hirsch2/juwels/plankton-dinov2/dinov2/train/train.py \
     --no-resume \
     --config-file dinov2/configs/train/vits14.yaml \
     --run_name=${SLURM_JOB_ID}_${N_GPUS_TOTAL}_gpu_pre \
     train.output_dir='/p/project1/hai_aqqua/hirsch2/dinov2_output' \
     train.use_torch_compile=true \
     train.dataset_path=SimpleLMDB:split=TRAIN:root=/p/project1/hai_aqqua/hirsch2/data/ImageNet1k_lmdb:extra=None \
     train.num_workers=$N_CPUS \
     train.batch_size_per_gpu=$BATCH_S \
     student.pretrained_weights="/p/project1/hai_aqqua/hirsch2/dinov2_vits14_pretrain.pth"
