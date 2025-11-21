# REPLUG LSR: Training BGE Retriever with LLM Supervision

This repository contains the implementation of REPLUG LSR (Retrieval-Augmented Language Model with LM-Supervised Retrieval) adapted for BGE retriever and Mistral/Llama2 language models.

## Overview

REPLUG LSR trains a dense retriever (BGE) to retrieve documents that help a frozen language model (Mistral/Llama2) make better predictions. The key idea is to use the LM's perplexity as supervision signal to adapt the retriever.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    REPLUG LSR Training                   │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Query (128 tokens) ──► BGE Retriever (Trainable)       │
│                              │                            │
│                              ├──► Top-K Documents        │
│                              │                            │
│  ┌──────────────────────────┴────────────────┐          │
│  │                                             │          │
│  ▼                                             ▼          │
│  Doc1 + Query + Continuation ──► LM (Frozen) ──► Score1 │
│  Doc2 + Query + Continuation ──► LM (Frozen) ──► Score2 │
│  ...                                            ...       │
│  DocK + Query + Continuation ──► LM (Frozen) ──► ScoreK │
│                                                           │
│  Retrieval Likelihood P_R(d|x) ──┐                      │
│                                    │                      │
│                                    ├──► KL Divergence    │
│                                    │      Loss           │
│  LM Likelihood Q_LM(d|x,y) ───────┘                     │
│  (from perplexity scores)                                │
│                                                           │
│  Loss = KL(P_R || Q_LM)                                  │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## Files

1. **preprocess_data.py**: Preprocesses wikitext-103-raw-v1 dataset
2. **train_replug.py**: Main training script for REPLUG LSR
3. **run_preprocess.sh**: Shell script to run preprocessing
4. **run_train.sh**: Shell script to run training

## Prerequisites

```bash
pip install torch transformers accelerate datasets faiss-gpu FlagEmbedding tqdm
```

## Usage

### Step 1: Data Preprocessing

First, preprocess the wikitext-103-raw-v1 dataset to create training samples:

```bash
chmod +x run_preprocess.sh
./run_preprocess.sh
```

**Configuration via environment variables:**

```bash
export DATASET_PATH="~/weirubin/datasets/wikitext"
export DATASET_NAME="wikitext-103-raw-v1"
export TOKENIZER_PATH="mistralai/Mistral-7B-v0.1"  # or meta-llama/Llama-2-7b-hf
export OUTPUT_DIR="./data/replug_wikitext_processed"
export QUERY_LENGTH=128
export CONTINUATION_LENGTH=128

./run_preprocess.sh
```

**Or run directly:**

```bash
python preprocess_data.py \
    --dataset_path ~/weirubin/datasets/wikitext \
    --dataset_name wikitext-103-raw-v1 \
    --tokenizer_name_or_path mistralai/Mistral-7B-v0.1 \
    --output_dir ./data/replug_wikitext_processed \
    --query_length 128 \
    --continuation_length 128 \
    --min_length 256
```

**What this does:**
- Loads wikitext-103-raw-v1 train split
- Tokenizes all text using the LLM tokenizer
- Finds sequences with ≥256 tokens
- Creates samples: first 128 tokens = query, next 128 tokens = continuation
- Saves processed dataset to disk

### Step 2: Train REPLUG LSR

Train the BGE retriever using LLM supervision:

```bash
chmod +x run_train.sh
./run_train.sh
```

**Configuration via environment variables:**

```bash
# Model paths
export RETRIEVER_MODEL="/mnt/petrelfs/linzhouhan/jqcao/projects/REPLUG/bge/models/bge-base-en-v1.5"
export LM_MODEL="mistralai/Mistral-7B-v0.1"  # or meta-llama/Llama-2-7b-hf

# Data paths
export DATASET_PATH="./data/replug_wikitext_processed"
export CORPUS_PATH="/mnt/petrelfs/linzhouhan/jqcao/projects/REPLUG/data/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl"
export INDEX_DIR="/mnt/petrelfs/linzhouhan/jqcao/projects/REPLUG/bge/wiki_rag_2048ncentroid_index"

# Output
export OUTPUT_DIR="./outputs/replug_lsr_trained"

# Training hyperparameters
export NUM_EPOCHS=3
export BATCH_SIZE=4
export GRADIENT_ACCUM=8
export LEARNING_RATE=2e-5

./run_train.sh
```

**Or run directly with accelerate:**

```bash
accelerate launch train_replug.py \
    --retriever_model_path /path/to/bge-base-en-v1.5 \
    --lm_model_path mistralai/Mistral-7B-v0.1 \
    --dataset_path ./data/replug_wikitext_processed \
    --corpus_path /path/to/wiki/corpus.jsonl \
    --index_dir /path/to/bge/index \
    --output_dir ./outputs/replug_lsr_trained \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --top_k 20 \
    --retrieved_max_length 128 \
    --temperature_retrieval 0.1 \
    --temperature_lm 0.1 \
    --checkpointing_steps 3000 \
    --logging_steps 10 \
    --with_tracking \
    --report_to tensorboard
```

### Step 3: Multi-GPU Training

For multi-GPU training, configure accelerate:

```bash
accelerate config
```

Then run the training script as above. Accelerate will automatically handle distributed training.

## Training Process

### What Happens During Training:

1. **For each batch:**
   - Decode query and continuation from token IDs
   - Retrieve top-K documents using BGE retriever
   - For each retrieved document:
     - Concatenate: `document + query + continuation`
     - Compute log probability of continuation using frozen LM
   - Compute LM likelihood: `Q(d | x, y) = softmax(log_probs / temperature_lm)`
   - Compute retrieval likelihood: `P_R(d | x) = softmax(scores / temperature_retrieval)`
   - Compute KL divergence: `Loss = KL(P_R || Q_LM)`

2. **Optimization:**
   - Backpropagate loss through BGE retriever (LM frozen)
   - Update retriever parameters to make it retrieve documents that lower LM perplexity

3. **Periodic refresh:**
   - Every `index_refresh_steps`, re-encode corpus with updated retriever
   - Rebuild FAISS index (optional, currently commented out)

## Key Hyperparameters

### Training Parameters:
- `num_train_epochs`: Number of training epochs (default: 3)
- `per_device_train_batch_size`: Batch size per GPU (default: 4)
- `gradient_accumulation_steps`: Gradient accumulation (default: 8)
- `learning_rate`: Learning rate (default: 2e-5)
- `warmup_ratio`: Warmup ratio (default: 0.1)
- `max_grad_norm`: Gradient clipping (default: 1.0)

### Retrieval Parameters:
- `top_k`: Number of documents to retrieve (default: 20)
- `retrieved_max_length`: Max tokens per document (default: 128)

### LSR Parameters:
- `temperature_retrieval`: Temperature for retrieval softmax (default: 0.1)
- `temperature_lm`: Temperature for LM softmax (default: 0.1)

Lower temperature = sharper distribution (more focused on top documents)
Higher temperature = flatter distribution (more exploration)

## Expected Results

Based on the REPLUG paper:
- **Language Modeling**: 6-8% perplexity reduction on various LMs
- **Convergence**: Usually converges within 25k steps (~3 epochs)
- **Memory**: ~40GB GPU memory for Mistral-7B + BGE-base

## Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir ./outputs/replug_lsr_trained
```

Metrics logged:
- `train_loss`: KL divergence loss
- `learning_rate`: Current learning rate
- `epoch`: Current epoch

## Checkpoints

Checkpoints saved at:
- `./outputs/replug_lsr_trained/checkpoint-{step}/`: Intermediate checkpoints
- `./outputs/replug_lsr_trained/final/`: Final trained model

## Using the Trained Retriever

After training, use the trained retriever with your BGE system:

```python
from FlagEmbedding import FlagModel

# Load trained retriever
trained_retriever = FlagModel(
    "./outputs/replug_lsr_trained/final",
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
    use_fp16=True
)

# Use for retrieval
query_embedding = trained_retriever.encode_queries(["your query"], convert_to_numpy=True)
# ... continue with FAISS search
```

## Differences from Original REPLUG

1. **Retriever**: Uses BGE instead of Contriever
2. **LM**: Uses Mistral/Llama2 instead of GPT-3
3. **Framework**: Uses Accelerate instead of custom distributed training
4. **Index**: Uses your pre-built BGE index instead of building from scratch

## Troubleshooting

### Out of Memory:
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Use smaller `top_k` value
- Reduce `retrieved_max_length`

### Slow Training:
- Increase `per_device_train_batch_size` if memory allows
- Use multiple GPUs with accelerate
- Reduce `top_k` to process fewer documents

### Poor Performance:
- Try different `temperature_retrieval` and `temperature_lm` values
- Increase training epochs
- Use more training data
- Verify index quality

## Citation

If you use this code, please cite the original REPLUG paper:

```bibtex
@article{shi2023replug,
  title={REPLUG: Retrieval-Augmented Black-Box Language Models},
  author={Shi, Weijia and Min, Sewon and Yasunaga, Michihiro and Seo, Minjoon and James, Rich and Lewis, Mike and Zettlemoyer, Luke and Yih, Wen-tau},
  journal={arXiv preprint arXiv:2301.12652},
  year={2023}
}
```

## License

This implementation is provided for research purposes. Please refer to the original REPLUG paper and BGE model licenses.