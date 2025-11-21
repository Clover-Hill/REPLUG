#!/bin/bash
# SBATCH --job-name=jqcao_replug
# SBATCH --partition=plm
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=1
# SBATCH --gres=gpu:8
# SBATCH --cpus-per-task=96
# SBATCH --exclusive
# SBATCH --output=./logs/replug_%x.log

# Model paths
RETRIEVER_MODEL=/mnt/petrelfs/linzhouhan/jqcao/projects/REPLUG/bge/models/bge-base-en-v1.5
LM_MODEL=/mnt/petrelfs/linzhouhan/weirubin/models/Mistral-7B-v0.3

# Data paths
DATASET_PATH=./data/replug_wikitext_mistral
CORPUS_PATH=/mnt/petrelfs/linzhouhan/jqcao/projects/REPLUG/data/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl
INDEX_DIR=/mnt/petrelfs/linzhouhan/jqcao/projects/REPLUG/bge/wiki_rag_2048ncentroid_index

# Output
OUTPUT_DIR=./checkpoint/replug_mistral_bge

# Training hyperparameters
NUM_EPOCHS=1
BATCH_SIZE=4
GRADIENT_ACCUM=4
LEARNING_RATE=2e-5
WARMUP_RATIO=0.1

# Retrieval parameters
TOP_K=10
RETRIEVED_MAX_LENGTH=128

# LSR parameters
# testing
TEMP_RETRIEVAL=0.1
TEMP_LM=0.1

# Checkpointing
CHECKPOINT_STEPS=3000
LOGGING_STEPS=1
INDEX_REFRESH_STEPS=3000

echo "=========================================="
echo "REPLUG LSR Training"
echo "=========================================="
echo "Retriever: $RETRIEVER_MODEL"
echo "LM: $LM_MODEL"
echo "Dataset: $DATASET_PATH"
echo "Index: $INDEX_DIR"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo "Training Configuration:"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRADIENT_ACCUM"
echo "  Learning rate: $LEARNING_RATE"
echo "  Top-K retrieval: $TOP_K"
echo "=========================================="

wandb login 9da2723a6023bbc4dd0110a28a710da214748a3a
WANDB_PROJECT="REPLUG"

# Run training with accelerate
accelerate launch train_replug.py \
    --retriever_model_path "$RETRIEVER_MODEL" \
    --lm_model_path "$LM_MODEL" \
    --dataset_path "$DATASET_PATH" \
    --corpus_path "$CORPUS_PATH" \
    --index_dir "$INDEX_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUM" \
    --learning_rate "$LEARNING_RATE" \
    --warmup_ratio "$WARMUP_RATIO" \
    --top_k "$TOP_K" \
    --retrieved_max_length "$RETRIEVED_MAX_LENGTH" \
    --temperature_retrieval "$TEMP_RETRIEVAL" \
    --temperature_lm "$TEMP_LM" \
    --checkpointing_steps "$CHECKPOINT_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --index_refresh_steps "$INDEX_REFRESH_STEPS" \
    --with_tracking \
    --report_to wandb

echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"