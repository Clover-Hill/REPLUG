"""
Preprocess wikitext-103-raw-v1 for REPLUG training.
Tokenizes the train split and creates samples with:
- First 128 tokens as query
- Next 128 tokens as continuation
"""

import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess wikitext data for REPLUG training")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="~/weirubin/datasets/wikitext",
        help="Path to wikitext dataset directory"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext-103-raw-v1",
        help="Wikitext dataset name"
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        required=True,
        help="Path to tokenizer (e.g., mistralai/Mistral-7B-v0.1)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save processed dataset"
    )
    parser.add_argument(
        "--query_length",
        type=int,
        default=128,
        help="Length of query tokens"
    )
    parser.add_argument(
        "--continuation_length",
        type=int,
        default=128,
        help="Length of continuation tokens"
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=256,
        help="Minimum length required (query_length + continuation_length)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate parameters
    assert args.min_length == args.query_length + args.continuation_length, \
        f"min_length ({args.min_length}) should equal query_length + continuation_length ({args.query_length + args.continuation_length})"
    
    logger.info(f"Loading dataset from {args.dataset_path}")
    
    # Load dataset
    dataset = load_dataset(
        args.dataset_path,
        args.dataset_name,
        split="train"
    )
    
    logger.info(f"Loaded {len(dataset)} samples from train split")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Process dataset
    logger.info("Tokenizing and filtering dataset...")
    
    processed_samples = []
    
    for example in tqdm(dataset, desc="Processing samples"):
        text = example["text"]
        
        # Skip empty or very short texts
        if not text or len(text.strip()) < 10:
            continue
        
        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Check if we have enough tokens
        if len(tokens) < args.min_length:
            continue
        
        # Create samples with sliding window
        # We can create multiple samples from one long text
        for i in range(0, len(tokens) - args.min_length + 1, args.min_length):
            query_tokens = tokens[i:i + args.query_length]
            continuation_tokens = tokens[i + args.query_length:i + args.min_length]
            
            # Ensure we have exactly the right lengths
            if len(query_tokens) != args.query_length or len(continuation_tokens) != args.continuation_length:
                continue
            
            processed_samples.append({
                "query_input_ids": query_tokens,
                "continuation_input_ids": continuation_tokens,
                "input_ids": query_tokens + continuation_tokens,  # Full sequence
            })
    
    logger.info(f"Created {len(processed_samples)} training samples")
    
    # Convert to dataset
    from datasets import Dataset as HFDataset
    processed_dataset = HFDataset.from_list(processed_samples)
    
    # Save to disk
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving processed dataset to {output_path}")
    processed_dataset.save_to_disk(str(output_path))
    
    # Save metadata
    metadata = {
        "num_samples": len(processed_samples),
        "query_length": args.query_length,
        "continuation_length": args.continuation_length,
        "tokenizer": args.tokenizer_name_or_path,
        "source_dataset": f"{args.dataset_path}/{args.dataset_name}",
    }
    
    import json
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Preprocessing complete!")
    logger.info(f"Total samples: {len(processed_samples)}")
    logger.info(f"Query length: {args.query_length} tokens")
    logger.info(f"Continuation length: {args.continuation_length} tokens")

if __name__ == "__main__":
    main()