#!/bin/bash

# Data Preprocessing Script for REPLUG LSR
# This script preprocesses wikitext-103-raw-v1 for training

set -e

# Configuration
DATASET_PATH=/mnt/petrelfs/linzhouhan/weirubin/datasets
DATASET_NAME=wikitext-103-raw-v1
TOKENIZER_PATH=/mnt/petrelfs/linzhouhan/weirubin/models/Mistral-7B-v0.3
OUTPUT_DIR=./data/replug_wikitext_mistral
QUERY_LENGTH=128
CONTINUATION_LENGTH=128

echo "=========================================="
echo "REPLUG Data Preprocessing"
echo "=========================================="
echo "Dataset: $DATASET_NAME"
echo "Tokenizer: $TOKENIZER_PATH"
echo "Output: $OUTPUT_DIR"
echo "Query length: $QUERY_LENGTH"
echo "Continuation length: $CONTINUATION_LENGTH"
echo "=========================================="

# Run preprocessing
python preprocess_data.py \
    --dataset_path "$DATASET_PATH" \
    --dataset_name "$DATASET_NAME" \
    --tokenizer_name_or_path "$TOKENIZER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --query_length "$QUERY_LENGTH" \
    --continuation_length "$CONTINUATION_LENGTH" \
    --min_length 256

echo "Preprocessing complete!"
echo "Processed dataset saved to: $OUTPUT_DIR"