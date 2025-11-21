#!/bin/bash

# Configuration
MODEL_PATH=/mnt/petrelfs/linzhouhan/weirubin/models/Mistral-7B-v0.3
MODEL_TYPE=mistral
DATASETS=("webqa")
BASE_RESULTS_DIR=/mnt/petrelfs/linzhouhan/jqcao/projects/REPLUG/results
INDEX_DIR="/mnt/petrelfs/linzhouhan/jqcao/projects/REPLUG/bge/wiki_rag_2048ncentroid_index"
TOP_K=5
NPROBE_VALUES=(32)

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=========================================="
echo "Starting evaluation pipeline for multiple datasets"
echo "Model: ${MODEL_PATH}"
echo "Datasets: ${DATASETS[@]}"
echo "NPROBE values: ${NPROBE_VALUES[@]}"
echo "TOP_K: ${TOP_K}"
echo "Timestamp: ${TIMESTAMP}"
echo "=========================================="

# Create summary log file
SUMMARY_LOG="logs/evaluation_summary_${TIMESTAMP}.log"
mkdir -p logs

# Track total experiments
TOTAL_EXPERIMENTS=$((${#DATASETS[@]} * ${#NPROBE_VALUES[@]}))
CURRENT_EXPERIMENT=0

# Main evaluation loop
for DATA in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Evaluating dataset: ${DATA}"
    echo "=========================================="
    
    # Loop over NPROBE values
    for NPROBE in "${NPROBE_VALUES[@]}"; do
        CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))
        
        echo ""
        echo "------------------------------------------"
        echo "Experiment ${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}"
        echo "Dataset: ${DATA}, NPROBE: ${NPROBE}"
        echo "------------------------------------------"
        
        # Set method type and results directory
        METHOD_TYPE=rag_bge_nprobe${NPROBE}_top${TOP_K}
        RESULTS_DIR=${BASE_RESULTS_DIR}/${DATA}/${MODEL_TYPE}/${METHOD_TYPE}/
        
        # Create log file for this specific run
        LOG_FILE="logs/${DATA}_${MODEL_TYPE}_nprobe${NPROBE}_${TIMESTAMP}.log"
        
        echo "Results directory: ${RESULTS_DIR}"
        echo "Log file: ${LOG_FILE}"
        echo "Running RAG evaluation with NPROBE=${NPROBE}..."
        
        # Record start time
        START_TIME=$(date +%s)
        
        # Run evaluation
        CUDA_VISIBLE_DEVICES=0 python -m rag_eval.run_eval_bge \
            --data ${DATA} \
            --model_name_or_path ${MODEL_PATH} \
            --results_dir ${RESULTS_DIR} \
            --model_type ${MODEL_TYPE} \
            --use_rag \
            --nprobe ${NPROBE} \
            --top_k ${TOP_K} \
            --index_dir ${INDEX_DIR} \
            2>&1 | tee -a ${LOG_FILE}
        
        # Check execution status
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        if [ ${EXIT_CODE} -eq 0 ]; then
            echo "✓ RAG evaluation completed successfully"
            echo "✓ Dataset: ${DATA}, NPROBE: ${NPROBE}, Duration: ${DURATION}s" >> ${SUMMARY_LOG}
            echo "  Results saved to: ${RESULTS_DIR}"
            echo "  Duration: ${DURATION} seconds"
        else
            echo "✗ RAG evaluation failed"
            echo "✗ Dataset: ${DATA}, NPROBE: ${NPROBE}, Failed after ${DURATION}s" >> ${SUMMARY_LOG}
            echo "  Error details in: ${LOG_FILE}"
            
            # Ask whether to continue or stop
            read -p "Continue with next experiment? (y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Stopping evaluation pipeline"
                exit 1
            fi
        fi
        
        echo "------------------------------------------"
        echo "Completed: ${DATA} with NPROBE=${NPROBE}"
        echo ""
    done
    
    echo "=========================================="
    echo "Completed all NPROBE evaluations for dataset: ${DATA}"
    echo "=========================================="
done

# Final summary
echo ""
echo "=========================================="
echo "EVALUATION PIPELINE COMPLETED"
echo "=========================================="
echo "Summary:"
echo "  - Model: ${MODEL_TYPE}"
echo "  - Model Path: ${MODEL_PATH}"
echo "  - Datasets evaluated: ${DATASETS[@]}"
echo "  - NPROBE values tested: ${NPROBE_VALUES[@]}"
echo "  - TOP_K: ${TOP_K}"
echo "  - Total experiments: ${TOTAL_EXPERIMENTS}"
echo "  - Summary log: ${SUMMARY_LOG}"
echo ""
echo "Results structure:"
for DATA in "${DATASETS[@]}"; do
    echo "  ${DATA}:"
    for NPROBE in "${NPROBE_VALUES[@]}"; do
        METHOD_TYPE=rag_bge_nprobe${NPROBE}_top${TOP_K}
        echo "    └── ${BASE_RESULTS_DIR}/${DATA}/${MODEL_TYPE}/${METHOD_TYPE}/"
    done
done
echo "=========================================="

# Display summary log
if [ -f ${SUMMARY_LOG} ]; then
    echo ""
    echo "Execution Summary:"
    cat ${SUMMARY_LOG}
fi