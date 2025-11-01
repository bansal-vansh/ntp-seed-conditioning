#!/bin/bash
START_DEVICE=1
MAX_PARALLEL=8
GPU_COUNT=8

NODE_VOCAB_SIZE=15
NUM_NODES=9
SEED_VOCAB_SIZE=26
SEED_LENGTHS=(0 5 20)
# SEED_PER_PI="-seed_per_pi"
# SEED_PER_PI=""

NUM_TRAIN_SAMPLES=10000
EPOCHS=200
BATCH_SIZE=64
GRAD_ACC_STEPS=1
MODEL_TYPE="gpt2"
TOKENIZER="default"  # options: default, custom

NUM_EVAL_SAMPLES=1000
NUM_EVAL_RUNS=3
NUM_CKPTS=100
TOP_P=1.0

LR=5e-5
RNG=20

SAVE_NAME="M${NODE_VOCAB_SIZE}-N${NUM_NODES}-H${SEED_VOCAB_SIZE}-NT${NUM_TRAIN_SAMPLES}-top_p${TOP_P}-${MODEL_TYPE}-BS$((BATCH_SIZE * GRAD_ACC_STEPS))-LR${LR}-E${EPOCHS}-RNG${RNG}-TOK${TOKENIZER}"
# Function to run a single job
run_job() {
  SEED_LEN=$1
  i=$2
  LOG_FILE="/datastor1/vansh/lang_sampling/logs/circle/${SAVE_NAME}/HL${SEED_LEN}.out"
  mkdir -p "$(dirname "$LOG_FILE")"

  DEVICE=$((i % GPU_COUNT))
  echo "Starting seed_length=$SEED_LEN on CUDA:$DEVICE → $LOG_FILE"
  
  export PYTHONPATH=$(pwd)
  CUDA_VISIBLE_DEVICES=$DEVICE python main/circle.py \
    --M "$NODE_VOCAB_SIZE" \
    --N "$NUM_NODES" \
    --H "$SEED_VOCAB_SIZE" \
    --HL "$SEED_LEN" \
    --epochs "$EPOCHS" \
    --model_type "$MODEL_TYPE" \
    --save_name "$SAVE_NAME" \
    --eval_runs "$NUM_EVAL_RUNS" \
    --num_train_samples "$NUM_TRAIN_SAMPLES" \
    --num_eval_samples "$NUM_EVAL_SAMPLES" \
    --top_p "$TOP_P" \
    --learning_rate "$LR" \
    --seed "$RNG" \
    --batch_size "$BATCH_SIZE" \
    --grad_acc_steps "$GRAD_ACC_STEPS" \
    --num_ckpts "$NUM_CKPTS" \
    --add_new_tokens \
    --custom_tokenizer \
    --device 0 > "$LOG_FILE" 2>&1 &
}

# Main loop
i=$START_DEVICE
for SEED_LEN in "${SEED_LENGTHS[@]}"; do
  run_job "$SEED_LEN" "$i"
  ((i++))

  # Limit to MAX_PARALLEL concurrent jobs
  if (( i % MAX_PARALLEL == 0 )); then
    wait
  fi
done

# Final wait to ensure all jobs finish
wait
echo "✅ All jobs complete."