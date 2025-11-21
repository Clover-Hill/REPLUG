"""
REPLUG LSR Training Script
Trains BGE retriever to accommodate LLM preferences using KL divergence loss
"""

import sys
import os
import argparse
import logging
import math
from pathlib import Path
from tqdm import tqdm
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
)
from FlagEmbedding import FlagModel

# Import BGE retrieval system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train BGE retriever using REPLUG LSR")
    
    # Model paths
    parser.add_argument(
        "--retriever_model_path",
        type=str,
        required=True,
        help="Path to BGE retriever model"
    )
    parser.add_argument(
        "--lm_model_path",
        type=str,
        required=True,
        help="Path to LLM (Mistral/Llama2) for supervision"
    )
    
    # Data paths
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to preprocessed dataset"
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        required=True,
        help="Path to corpus for retrieval (e.g., wiki jsonl)"
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        required=True,
        help="Path to BGE index directory"
    )
    
    # Training parameters
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    # Retrieval parameters
    parser.add_argument("--top_k", type=int, default=20, help="Number of documents to retrieve")
    parser.add_argument("--retrieved_max_length", type=int, default=128)
    
    # LSR parameters
    parser.add_argument("--temperature_retrieval", type=float, default=0.1, help="Temperature for retrieval likelihood")
    parser.add_argument("--temperature_lm", type=float, default=0.1, help="Temperature for LM likelihood")
    
    # Checkpoint and logging
    parser.add_argument("--checkpointing_steps", type=int, default=3000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--index_refresh_steps", type=int, default=3000, help="Steps to refresh document embeddings")
    
    # Misc
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    
    args = parser.parse_args()
    return args


class BGERetriever:
    """Wrapper for BGE retrieval system"""
    
    def __init__(self, model_path, index_dir, corpus_path, top_k=20):
        self.model = FlagModel(
            model_path,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
            use_fp16=True
        )
        self.top_k = top_k
        
        # Load index and metadata
        import faiss
        import pickle
        
        index_path = os.path.join(index_dir, "faiss.index")
        metadata_path = os.path.join(index_dir, "metadata.pkl")
        
        logger.info(f"Loading index from {index_path}")
        self.index = faiss.read_index(index_path)
        
        logger.info(f"Loading metadata from {metadata_path}")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            self.passages = metadata['passages']
            self.passage_ids = metadata['passage_ids']
        
        logger.info(f"Loaded {len(self.passages)} passages")
    
    def retrieve(self, queries, top_k=None):
        """Retrieve top-k documents for each query"""
        if top_k is None:
            top_k = self.top_k
        
        # Encode queries
        query_embeddings = self.model.encode_queries(queries, convert_to_numpy=True)
        query_embeddings = query_embeddings.astype(np.float32)
        
        # Normalize
        import faiss
        faiss.normalize_L2(query_embeddings)
        
        # Search
        scores, indices = self.index.search(query_embeddings, top_k)
        
        # Format results
        results = []
        for batch_idx in range(len(queries)):
            batch_results = []
            for idx, score in zip(indices[batch_idx], scores[batch_idx]):
                if idx < len(self.passages):
                    batch_results.append({
                        'text': self.passages[idx],
                        'score': float(score),
                        'id': self.passage_ids[idx]
                    })
            results.append(batch_results)
        
        return results
    
    def compute_retrieval_scores(self, queries):
        """Compute retrieval scores for queries"""
        # Encode queries
        query_embeddings = self.model.encode_queries(queries, convert_to_numpy=True)
        return torch.from_numpy(query_embeddings)
    
    def update_index(self, index_dir):
        """Refresh the index with updated embeddings"""
        import faiss
        import pickle
        
        # Reload index
        index_path = os.path.join(index_dir, "faiss.index")
        self.index = faiss.read_index(index_path)
        
        logger.info(f"Index refreshed from {index_path}")


def collate_fn(batch):
    """Custom collate function"""
    query_input_ids = torch.stack([torch.tensor(item["query_input_ids"]) for item in batch])
    continuation_input_ids = torch.stack([torch.tensor(item["continuation_input_ids"]) for item in batch])
    input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
    
    return {
        "query_input_ids": query_input_ids,
        "continuation_input_ids": continuation_input_ids,
        "input_ids": input_ids,
    }


def compute_lm_likelihood(lm_model, tokenizer, retrieved_docs, queries, continuations, 
                         temperature, retrieved_max_length, device):
    """
    Compute LM likelihood Q(d | x, y) for each retrieved document
    
    Args:
        lm_model: Language model
        tokenizer: Tokenizer
        retrieved_docs: List of lists of retrieved documents
        queries: Query texts
        continuations: Continuation texts
        temperature: Temperature for softmax
        retrieved_max_length: Max length for retrieved documents
        device: Device
    
    Returns:
        LM likelihood distribution over documents
    """
    batch_size = len(queries)
    num_docs = len(retrieved_docs[0])
    
    all_log_probs = []
    
    # Process each sample in the batch
    for i in range(batch_size):
        sample_log_probs = []
        
        # For each retrieved document
        for doc in retrieved_docs[i]:
            doc_text = doc['text']
            query_text = queries[i]
            continuation_text = continuations[i]
            
            # Tokenize document (truncate if needed)
            doc_tokens = tokenizer.encode(doc_text, add_special_tokens=False)[:retrieved_max_length]
            
            # Tokenize query
            query_tokens = tokenizer.encode(query_text, add_special_tokens=False)
            
            # Tokenize continuation
            continuation_tokens = tokenizer.encode(continuation_text, add_special_tokens=False)
            
            # Concatenate: doc + query
            input_tokens = doc_tokens + query_tokens
            input_ids = torch.tensor([input_tokens]).to(device)
            
            # Target tokens (continuation)
            target_ids = torch.tensor([continuation_tokens]).to(device)
            
            # Forward pass through LM
            with torch.no_grad():
                outputs = lm_model(input_ids)
                logits = outputs.logits
                
                # Get logits for continuation positions
                # logits shape: [1, seq_len, vocab_size]
                continuation_logits = logits[:, -len(continuation_tokens):, :]
                
                # Compute log probability of continuation
                log_probs = F.log_softmax(continuation_logits, dim=-1)
                
                # Gather log probs for target tokens
                target_log_probs = torch.gather(
                    log_probs,
                    dim=-1,
                    index=target_ids.unsqueeze(-1)
                ).squeeze(-1)
                
                # Sum log probs for the continuation
                total_log_prob = target_log_probs.sum().item()
                
                sample_log_probs.append(total_log_prob)
        
        all_log_probs.append(sample_log_probs)
    
    # Convert to tensor [batch_size, num_docs]
    all_log_probs = torch.tensor(all_log_probs).to(device)
    
    # Apply temperature and softmax to get Q(d | x, y)
    lm_likelihood = F.softmax(all_log_probs / temperature, dim=-1)
    
    return lm_likelihood


def compute_retrieval_likelihood(retriever_model, queries, retrieved_docs, temperature, device):
    """
    Compute retrieval likelihood P_R(d | x)
    
    Args:
        retriever_model: BGE retriever model
        queries: Query texts
        retrieved_docs: List of lists of retrieved documents with scores
        temperature: Temperature for softmax
        device: Device
    
    Returns:
        Retrieval likelihood distribution
    """
    batch_size = len(queries)
    
    # Extract scores from retrieved docs
    scores = []
    for i in range(batch_size):
        sample_scores = [doc['score'] for doc in retrieved_docs[i]]
        scores.append(sample_scores)
    
    # Convert to tensor [batch_size, num_docs]
    scores = torch.tensor(scores).to(device)
    
    # Apply temperature and softmax
    retrieval_likelihood = F.softmax(scores / temperature, dim=-1)
    
    return retrieval_likelihood


def main():
    args = parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to if args.with_tracking else None,
        project_dir=args.output_dir if args.with_tracking else None,
    )
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Logging
    logger.info(accelerator.state)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    
    # Create output directory
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.lm_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load LM (frozen, for supervision only)
    logger.info(f"Loading language model: {args.lm_model_path}")
    lm_model = AutoModelForCausalLM.from_pretrained(
        args.lm_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    lm_model.eval()
    for param in lm_model.parameters():
        param.requires_grad = False
    
    # Load retriever (trainable)
    logger.info(f"Loading retriever: {args.retriever_model_path}")
    retriever = BGERetriever(
        args.retriever_model_path,
        args.index_dir,
        args.corpus_path,
        top_k=args.top_k
    )
    
    # Make retriever model trainable
    retriever.model.model.train()
    for param in retriever.model.model.parameters():
        param.requires_grad = True
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    
    # Create dataloader
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        retriever.model.model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * args.warmup_ratio)
    
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )
    
    # Prepare with accelerator
    retriever.model.model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        retriever.model.model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Training info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    # Training loop
    global_step = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            
            # Decode queries and continuations
            queries = [tokenizer.decode(ids, skip_special_tokens=True) 
                      for ids in batch["query_input_ids"]]
            continuations = [tokenizer.decode(ids, skip_special_tokens=True) 
                           for ids in batch["continuation_input_ids"]]
            
            # Retrieve documents
            retrieved_docs = retriever.retrieve(queries, top_k=args.top_k)
            
            # Compute LM likelihood Q(d | x, y)
            lm_likelihood = compute_lm_likelihood(
                lm_model,
                tokenizer,
                retrieved_docs,
                queries,
                continuations,
                args.temperature_lm,
                args.retrieved_max_length,
                accelerator.device
            )
            
            # Compute retrieval likelihood P_R(d | x)
            retrieval_likelihood = compute_retrieval_likelihood(
                retriever.model,
                queries,
                retrieved_docs,
                args.temperature_retrieval,
                accelerator.device
            )
            
            # Compute KL divergence loss
            # KL(P_R || Q_LM) = sum(P_R * log(P_R / Q_LM))
            # For numerical stability, use log-space:
            # KL(P || Q) = sum(P * (log P - log Q))
            loss = F.kl_div(
                retrieval_likelihood.log(),
                lm_likelihood,
                reduction='batchmean',
                log_target=False
            )
            
            # Backward pass
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(retriever.model.model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                progress_bar.update(1)
                global_step += 1
                
                # Logging
                if global_step % args.logging_steps == 0:
                    logger.info(f"Step {global_step}: loss = {loss.item():.4f}, lr = {lr_scheduler.get_last_lr()[0]:.2e}")
                    
                    if args.with_tracking:
                        accelerator.log({
                            "train_loss": loss.item(),
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "epoch": epoch,
                        }, step=global_step)
                
                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    output_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save retriever model
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(retriever.model.model)
                        unwrapped_model.save_pretrained(
                            output_dir,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                        )
                        logger.info(f"Saved checkpoint to {output_dir}")
                
                # Refresh index
                if global_step % args.index_refresh_steps == 0 and global_step > 0:
                    logger.info("Refreshing document embeddings and index... Not Done HERE!")
                    # TODO: Implement index refresh
                    # This requires re-encoding all documents and rebuilding the index
                    # For now, we skip this for simplicity
                    pass
            
            if global_step >= max_train_steps:
                break
    
    # Save final model
    if accelerator.is_main_process:
        output_dir = Path(args.output_dir) / "final"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        unwrapped_model = accelerator.unwrap_model(retriever.model.model)
        unwrapped_model.save_pretrained(output_dir)
        
        # Save training args
        with open(output_dir / "training_args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        
        logger.info(f"Training complete! Final model saved to {output_dir}")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()