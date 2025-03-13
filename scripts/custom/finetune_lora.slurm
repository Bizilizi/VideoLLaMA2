#!/bin/bash
#SBATCH --job-name="VideoLLaMA-2-8gpu"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --partition=DEADLINE
#SBATCH --comment="ICCV Rebuttal"
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/zverev/logs/videollama2/8gpu/slurm-%j.out
#SBATCH --error=/storage/slurm/zverev/logs/videollama2/8gpu/slurm-%j.out
#SBATCH --nodelist=node3,node20,node19

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
echo "START TIME: $(date)"
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

source activate videollama2av
set -x

mkdir -p /storage/local/zverev
mkdir -p /tmp/vggsound

# copy the squashfs file to the local storage
rsync -av --progress --ignore-existing /home/wiss/zverev/datasets/vggsound.squashfs /storage/local/zverev/vggsound.squashfs

# mount the squashfs file
squashfuse /storage/local/zverev/vggsound.squashfs /tmp/vggsound

# cd to the main dir
cd ../..

# Environment Variables
ARG_WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))


if [ "$ARG_WORLD_SIZE" -gt 1 ]; then
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    ARG_MASTER_ADDR=$master_addr
    ARG_NPROC_PER_NODE=${SLURM_GPUS_PER_NODE:-${1:-4}}
else
    ARG_MASTER_ADDR="127.0.0.1"
    ARG_NPROC_PER_NODE=${SLURM_GPUS:-${1:-4}}
fi

ARG_MASTER_PORT=$((10000 + RANDOM % 20000))

echo "NODELIST="${SLURM_NODELIST}
echo "ARG_MASTER_ADDR=$ARG_MASTER_ADDR"
echo "ARG_MASTER_PORT=$ARG_MASTER_PORT"

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=128
LOCAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]

# Log Arguments
export WANDB_PROJECT=videollama2qwen2_downstream_sft

RUN_NAME=siglip_tcv35_7b_16f_lora
DATA_DIR=/tmp/vggsound/
OUTP_DIR=work_dirs

# same goes for `\$(hostname -s|tr -dc '0-9')` - we want it to interpolate at `srun` time
LAUNCHER="torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank \$SLURM_NODEID \
    "

# EDIT the path+name of the python script and whatever args it needs
PROGRAM="videollama2/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed scripts/zero3.json \
    --model_type videollama2_qwen2 \
    --model_path Qwen/Qwen2-7B-Instruct \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_projector_type stc_connector_v35 \
    --pretrain_mm_mlp_adapter DAMO-NLP-SG/VideoLLaMA2.1-7B-16F-Base/mm_projector.bin \
    --data_path   datasets/vggsound/vggsound_train.json \
    --data_folder ${DATA_DIR} \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio pad \
    --num_frames 16 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/finetune_${RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 99 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 24 \
    --report_to tensorboard \
    --run_name $RUN_NAME \
    "

export CMD="$LAUNCHER $PROGRAM"
echo $CMD

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
    "

srun $SRUN_ARGS bash -c "$CMD"

echo "END TIME: $(date)"