#!/bin/bash

MAX_PARALLEL=8
GPU_COUNT=8
NUM_CHILDREN=5
DEPTH=4
SEED_VOCAB_SIZE=26
NUM_TRAIN_SAMPLES=1000
SEED_LENGTHS=(0 5 10 20)

# Function to run a single job
run_job() {
  SEED_LEN=$1
  DEVICE=$2
  SAVE_NAME="K${NUM_CHILDREN}-L${DEPTH}-H${SEED_VOCAB_SIZE}-N${NUM_TRAIN_SAMPLES}"
  LOG_FILE="logs/${SAVE_NAME}/HL${SEED_LEN}.out"
  mkdir -p "$(dirname "$LOG_FILE")"
  
  echo "Starting seed_length=$SEED_LEN on CUDA:$DEVICE → $LOG_FILE"
  # Remap physical GPU $DEVICE to visible device 0 for this subprocess
  CUDA_VISIBLE_DEVICES=$DEVICE python main/tree.py \
    --K "$NUM_CHILDREN" \
    --L "$DEPTH" \
    --H "$SEED_VOCAB_SIZE" \
    --HL "$SEED_LEN" \
    --save_name "${SAVE_NAME}" \
    --num_train_samples "${NUM_TRAIN_SAMPLES}" \
    --device 0 > "$LOG_FILE" 2>&1 &

}

# Main loop
i=4
for SEED_LEN in "${SEED_LENGTHS[@]}"; do
  DEVICE=$((i % GPU_COUNT))
  run_job "$SEED_LEN" "$DEVICE"
  ((i++))

  # Limit to MAX_PARALLEL concurrent jobs
  if (( i % MAX_PARALLEL == 0 )); then
    wait
  fi
done

# Final wait to ensure all jobs finish
wait
echo "✅ All jobs complete."