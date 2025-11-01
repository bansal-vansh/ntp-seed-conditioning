#!/bin/bash
START_DEVICE=5
MAX_PARALLEL=8
GPU_COUNT=8

NUM_PARENTS=5
NUM_CHILDREN=2500
EDGE_PROB=1
SEED_VOCAB_SIZE=26
SEED_LENGTHS=(0 5 20)
TOKENIZER="default"  # options: default, custom

NUM_TRAIN_SAMPLES=10000
EPOCHS=70
BATCH_SIZE=64
MODEL_TYPE="gpt2"

NUM_EVAL_SAMPLES=1000
EVAL_BATCH_SIZE=64
NUM_EVAL_RUNS=3
NUM_CKPTS=100
TOP_P=1.0
LR=5e-5
RNG=20

SAVE_NAME="P${NUM_PARENTS}-C${NUM_CHILDREN}-prob${EDGE_PROB}-H${SEED_VOCAB_SIZE}-N${NUM_TRAIN_SAMPLES}-top_p${TOP_P}-${MODEL_TYPE}-BS${BATCH_SIZE}-LR${LR}-E${EPOCHS}-RNG${RNG}-TOK${TOKENIZER}"

# # Function to run a single job
run_job_plan() {
  SEED_LEN=$1
  i=$2
  PLANNING_LOG_FILE="/datastor1/vansh/lang_sampling/logs/sibling/${SAVE_NAME}-plan/HL${SEED_LEN}.out"
  mkdir -p "$(dirname "$PLANNING_LOG_FILE")"

  DEVICE=$((i % GPU_COUNT))
  echo "Starting seed_length=$SEED_LEN with planning on CUDA:$DEVICE → $PLANNING_LOG_FILE"
  # Remap physical GPU $DEVICE to visible device 0 for this subprocess
  export PYTHONPATH=$(pwd)
  CUDA_VISIBLE_DEVICES=$DEVICE python main/sibling.py \
    --P "$NUM_PARENTS" \
    --C "$NUM_CHILDREN" \
    --prob "$EDGE_PROB" \
    --H "$SEED_VOCAB_SIZE" \
    --HL "$SEED_LEN" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --top_p "$TOP_P" \
    --model_type "$MODEL_TYPE" \
    --save_name "${SAVE_NAME}-plan" \
    --eval_runs "$NUM_EVAL_RUNS" \
    --num_train_samples "$NUM_TRAIN_SAMPLES" \
    --num_eval_samples "$NUM_EVAL_SAMPLES" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --num_ckpts "$NUM_CKPTS" \
    --learning_rate "$LR" \
    --add_new_tokens \
    --seed "$RNG" \
    --device 0 > "$PLANNING_LOG_FILE" 2>&1 &
}

# Main loop
i=$START_DEVICE
for SEED_LEN in "${SEED_LENGTHS[@]}"; do
  run_job_plan "$SEED_LEN" "$i"
  ((i++))

  # Limit to MAX_PARALLEL concurrent jobs
  if (( i % MAX_PARALLEL == 0 )); then
    wait
  fi
  # run_job_no_plan "$SEED_LEN" "$i"
  # ((i++))

  # # Limit to MAX_PARALLEL concurrent jobs
  # if (( i % MAX_PARALLEL == 0 )); then
  #   wait
  # fi
done

# Final wait to ensure all jobs finish
wait
echo "✅ All jobs complete."

# Function to run a single job
# run_job_no_plan() {
#   SEED_LEN=$1
#   i=$2
#   NOPLANNING_LOG_FILE="logs/sibling/${SAVE_NAME}-no_plan/HL${SEED_LEN}.out"
#   mkdir -p "$(dirname "$NOPLANNING_LOG_FILE")"

#   DEVICE=$((i % GPU_COUNT))
#   echo "Starting seed_length=$SEED_LEN without planning on CUDA:$DEVICE → $NOPLANNING_LOG_FILE"
  
#   export PYTHONPATH=$(pwd)
#   CUDA_VISIBLE_DEVICES=$DEVICE python main/sibling.py \
#     --P "$NUM_PARENTS" \
#     --C "$NUM_CHILDREN" \
#     --prob "$EDGE_PROB" \
#     --H "$SEED_VOCAB_SIZE" \
#     --HL "$SEED_LEN" \
#     --no_planning \
#     --epochs "$EPOCHS"\
#     --top_p "$TOP_P" \
#     --model_type "$MODEL_TYPE" \
#     --save_name "${SAVE_NAME}-no_plan" \
#     --eval_runs "$NUM_EVAL_RUNS" \
#     --num_train_samples "$NUM_TRAIN_SAMPLES" \
#     --num_eval_samples "$NUM_EVAL_SAMPLES" \
#     --num_ckpts "$NUM_CKPTS" \
#     --device 0 > "$NOPLANNING_LOG_FILE" 2>&1 &
# }