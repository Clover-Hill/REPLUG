## built-in
import argparse,json,os
import time
## third party
from transformers import (
    MistralForCausalLM,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    MixtralForCausalLM,
)
import torch
import datasets
from tqdm import tqdm
import pandas as pd

from rag_eval.utils import (
    stop_sequences_criteria,
    get_substring_match_score,
    eval_fact_checking,
    eval_truthfulqa,
    keyword_extraction_with_tfidf,
)

# from downstream.contriever.build_faiss_index import ContrieverFAISSIndex
from bge.bge import RAGConfig, WikipediaRAG

import pickle

def get_jsonl(f):
    import json
    return [json.loads(x) for x in open(f).readlines()]

def load_contrieverFaiss(top_k=5, nprobe=32, index_dir=None):
    assert index_dir is not None,"index_dir is not set"
    config = RAGConfig(
        embedding_model = "/mnt/petrelfs/linzhouhan/jqcao/projects/REPLUG/bge/models/bge-base-en-v1.5",
        index_save_dir = args.index_dir,
        test_mode=False,  # 处理全部数据
        batch_size=1024,
        index_type="IVF",
        nprobe=nprobe,
        use_gpu=False,
        top_k=5
    )

    # Initialize RAG system
    searcher = WikipediaRAG(config)
    
    searcher.load_complete_index()

    print(f"top_k is {top_k}, nprobe is {nprobe}")
    return searcher

def create_prompt_with_mistral_chat_format(messages,tokenizer,*args,**kwargs):

    formatted_text = ""
    for message in messages:
        if message['role'] == 'user':
            formatted_text += message['content'] + "\n"
        elif message['role'] == 'assistant':
            formatted_text += message['content'] + "\n"
    return formatted_text.strip()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval_prefix",
        default='colbertv2'
    )
    parser.add_argument(
        "--tf_idf_topk",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--base_model",
    )
    parser.add_argument(
        "--enable_progress_bar",
        type=eval,
        default=True,
    )
    parser.add_argument(
        "--data",
    )
    parser.add_argument(
        "--index_dir",
        default=None,
    )
    parser.add_argument(
        "--model_name_or_path",
    )
    parser.add_argument(
        "--knn_generator_path",
    )
    parser.add_argument(
        "--eval_metrics",
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--retrieval_topk",
        type=int,
        default=[1],
        nargs='+',
    )
    parser.add_argument(
        "--use_rag",
        action='store_true',
    )
    parser.add_argument(
        "--use_lora",
        action='store_true',
    )
    parser.add_argument(
        "--use_neuralknn",
        action='store_true',
    )
    parser.add_argument(
        "--retrieval_embed_length",
        type=int,default=0,
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        help="for debug",
    )
    parser.add_argument(
        "--results_dir",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--chat_format",
        default='mistral',
    )
    parser.add_argument(
        "--lmbda",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--model_type",
        default='llama2',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=32,
    )
    args = parser.parse_args()

    ## post-process
    if args.data in ['nq','hotpotqa','triviaqa','webqa']:
        args.task_type = 'open_qa'
        args.eval_metrics = 'substring_match'
    elif args.data in ['truthfulqa']:
        args.task_type = 'open_qa'
        args.eval_metrics = 'truthfulqa_f1_rl'
    elif args.data in ['factkg']:
        args.task_type = 'fact_checking'
        args.eval_metrics = 'fact_checking_acc'
    
    args.retrieval_topk = [x-1 for x in args.retrieval_topk] ## rank starts from 1
    
    if args.chat_format is not None:
        args.chat_format = eval(f"create_prompt_with_{args.chat_format}_chat_format")    
    
    return args


QA_PROMPT = "Question: {question}?\n"
FECT_CHECKING_PROPMT = "Claim: {question}\n"
BACKGROUND_PROMPT_TEMPLATE = "Background: {background}\n\n"

PROMPT_TEMPLATES = {
    "open_qa":QA_PROMPT,
    'fact_checking':FECT_CHECKING_PROPMT,
}

import math

def estimate_flops_per_token(model, hidden_size, num_layers, vocab_size, seq_len):
    """
    粗略估算每个 token 的 FLOPs:
    - Attention: O(seq_len * hidden_size^2)
    - FFN: O(hidden_size^2)
    """
    # attention per layer: ~4 * hidden_size^2
    attn_flops = 4 * hidden_size * hidden_size
    # FFN per layer: ~8 * hidden_size^2
    ffn_flops = 8 * hidden_size * hidden_size
    per_layer_flops = attn_flops + ffn_flops
    total_flops = per_layer_flops * num_layers
    return total_flops  # 单个 token 的 FLOPs


def get_start_prompt(task_type,use_rag,sample=None):
    if task_type == 'open_qa':
        return {
            True: "Refer to the background document and answer the questions:",
            False:"Answer the questions:"
        }[use_rag]
    elif task_type == 'fact_checking':
        return {
            True: "Refer to the background document and verify the following claims with \"True\" or \"False\":",
            False:"Verify the following claims with \"True\" or \"False\":"
        }[use_rag]

@torch.no_grad()
def llm_for_open_generation(
    llm,llm_tokenizer,
    prompts,
    retrieval_embeds,
    batch_size = 4,
    enable_progress_bar = True,
):

    generated_answers = []
    total_test_number = len(prompts)
    device = llm.device
    batched_prompts = [prompts[idx:idx+batch_size] for idx in range(0,len(prompts),batch_size)]

    profile_info = {
        "cuda_time_per_batch": [],
        "tokens_per_batch": [],
        "total_tokens": 0,
    }
    
    if retrieval_embeds is not None:
        batched_retrieval_embeds = [retrieval_embeds[idx:idx+batch_size] for idx in range(0,len(retrieval_embeds),batch_size)]
        assert len(batched_prompts) == len(batched_retrieval_embeds)

    progress_bar = tqdm(range(total_test_number),ncols=60,disable= not enable_progress_bar)
    for batch_idx in range(len(batched_prompts)):
        prompt = batched_prompts[batch_idx]
        tokenized_propmt = llm_tokenizer(prompt,padding='longest',return_tensors='pt')
        input_ids = tokenized_propmt.input_ids.to(device)
        attention_mask = tokenized_propmt.attention_mask.to(device)
        stopping_criteria = stop_sequences_criteria(llm_tokenizer, input_ids.shape[1], input_ids.shape[0])
        retrieval_kwargs = {}
        if retrieval_embeds is not None:
            embeds = batched_retrieval_embeds[batch_idx]
            embeds = [x for y in embeds for x in y]
            embeds = torch.stack(embeds).to(device)
            retrieval_kwargs['retrieval_embeds'] = embeds
            stopping_criteria = stop_sequences_criteria(llm_tokenizer, 0, input_ids.shape[0])

        start_time = time.time()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        if llm.config.model_type == "mistral":
            ## actual computation
            generated_output = llm.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                stopping_criteria=stopping_criteria,
                do_sample=False,
                max_new_tokens=100,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                **retrieval_kwargs,
            )
        else:
            ## actual computation
            generated_output = llm.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                stopping_criteria=stopping_criteria,
                do_sample=False,
                temperature=1.0,      # ← 显式设置
                top_p=1.0,            # ← 显式设置
                max_new_tokens=100,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                **retrieval_kwargs,
            )
        
        end_time = time.time()
        ender.record()
        torch.cuda.synchronize()
        cuda_time_ms = starter.elapsed_time(ender)  # 毫秒
        gen_tokens = generated_output.shape[1] - input_ids.shape[1]

        elapsed = end_time - start_time
        profile_info["cuda_time_per_batch"].append(cuda_time_ms)
        profile_info["tokens_per_batch"].append(gen_tokens * input_ids.shape[0])  # batch_size * gen_len
        profile_info["total_tokens"] += gen_tokens

        print(
            f"Example done in {elapsed:.2f}s, generated {gen_tokens} tokens, "
            f"speed={gen_tokens/elapsed:.2f} tokens/s, "
            f"cuda_time_ms={cuda_time_ms:.2f} ms, "
        )

        ## because HF generate with inputs_embeds would not return prompt
        input_length = 0 if retrieval_kwargs else input_ids.shape[1]
        results = tokenizer.batch_decode(generated_output[:,input_length:],skip_special_tokens=False)
        generated_answers.extend(results)
        progress_bar.update(batch_size)

    generated_answers = [x.strip() for x in generated_answers]
    return generated_answers, profile_info

def format_one_example(
    sample,include_answer,use_rag,retrieval_embed_length,task_type,rag_searcher=None,args=None
):
    
    question   = sample['question']
    prompt_dict = dict(question=question)
    prompt = PROMPT_TEMPLATES[task_type].format_map(prompt_dict).strip()
    backgrounds = []

    if use_rag:
        # backgrounds = sample['background'] ## a list
        # results = rag_searcher.search(question, top_k=args.top_k)
        results = rag_searcher.search(question)
        context_parts = []
        for i, result in enumerate(results['retrieved_passages']):
            context_parts.append(f"{i+1}. {result['passage']}")
        # context_parts = []
        # for i, result in enumerate(results):
        #     context_parts.append(f"{i+1}. {result['text']}")
        context_text = "\n".join(context_parts)
        context_content = f"\n{context_text}\n"

        # backgrounds.append(rag_searcher.search(question, top_k = 1)[0]["text"])
        backgrounds.append(context_content)
        background_prompts = ""
        for background in backgrounds:
            background_prompts += background + " "
        background_prompts = background_prompts.strip()
        prompt = BACKGROUND_PROMPT_TEMPLATE.format_map(dict(background=background_prompts)) + prompt

    return prompt,backgrounds

def get_n_shot_prompt(dev_data,n_shot,task_type,use_rag=False,retrieval_embed_length=0,rag_searcher=None,args=None):
    assert n_shot >= 0,n_shot
    n_shot_prompt = []
    n_shot_background = []
    if dev_data is not None:
        n_shot_examples = dev_data[:n_shot]
        for example in n_shot_examples:
            prompt,background = format_one_example(example,include_answer=True,use_rag=use_rag,retrieval_embed_length=retrieval_embed_length,task_type=task_type,rag_searcher=rag_searcher,args=args)
            n_shot_prompt.append(prompt)
            n_shot_background.append(background)

    return n_shot_prompt,n_shot_background

def prepare_prompts(
    dev_data,test_data,task_type,tokenizer,
    n_shot = 0, use_rag = False,
    retrieval_embed_length=0,
    chat_format = None,
    rag_searcher = None,
    model_type = None,
    args = None
):
    splitter = "\n\n"
    prompts = []
    backgrounds = []
    original_n_shot = n_shot
    for idx,sample in enumerate(test_data):
        n_shot = original_n_shot
        while True:
            prompt_start  = get_start_prompt(task_type,use_rag=use_rag,sample=sample) 
            prompt_end,background    = format_one_example(
                sample,include_answer=False,use_rag=use_rag,retrieval_embed_length=retrieval_embed_length,task_type=task_type,rag_searcher=rag_searcher,args=args)
            if 'subject' not in sample.keys():
                n_shot_prompt,n_shot_background = get_n_shot_prompt(dev_data,n_shot=n_shot,use_rag=use_rag,retrieval_embed_length=retrieval_embed_length,task_type=task_type,rag_searcher=rag_searcher)
            else:
                ## select n-shot within the same subjects for MMLU
                dev_data_with_same_subjects = []
                for d in dev_data:
                    if d['subject'] == sample['subject']:
                        dev_data_with_same_subjects.append(d)
                assert len(dev_data_with_same_subjects)==5,sample['subject']
                n_shot_prompt,n_shot_background = get_n_shot_prompt(dev_data_with_same_subjects,n_shot=n_shot,use_rag=use_rag,retrieval_embed_length=retrieval_embed_length,task_type=task_type)
            
            if model_type == "llama2" and task_type == "open_qa":
                if use_rag:
                    prompt_start = "Refer to the background and answer these questions:"
                    splitter = "\n"
                else:
                    prompt_start = "Answer these questions:"
                    splitter = "\n"
            if n_shot_prompt:  
                prompt = prompt_start + splitter + splitter.join(n_shot_prompt) + splitter + prompt_end  
            else: 
                prompt = prompt_start + splitter + prompt_end

            if chat_format is not None:
                messages = [{"role": "user", "content": prompt}]
                if model_type == "llama2":
                    prompt = chat_format(messages, tokenizer) + "Answer:"
                else:
                    prompt = chat_format(messages, tokenizer) + " The answer is:"

            tokenized_prompt = tokenizer(prompt,truncation=False,add_special_tokens=False).input_ids

            if len(tokenized_prompt) > 2048 and n_shot >= 1:
                n_shot -= 1
            else:
                break
        
        prompts.append(prompt)
        backgrounds.append(background+n_shot_background)

    print("**"*20,"show one example","**"*20)
    print(prompts[0])
    print("**"*20,"show one example","**"*20)

    return prompts,backgrounds

def load_dataset(data, args):
    if data == "factkg":
        file_path = "/mnt/petrelfs/linzhouhan/weirubin/projects/neuralKNN/downstream/data/factkg/factkg_test.pickle"
        with open(file_path, 'rb') as f:
            raw_data = pickle.load(f)  # dict: {claim: {...}}
        
        test_data = []
        for claim, meta in raw_data.items():
            # label 里可能是 list，要取第一个
            label = meta["Label"][0] if isinstance(meta["Label"], list) else meta["Label"]
            test_data.append({
                "id": claim,
                "question": claim,   # prompt 用 claim
                "answer": "True" if label else "False",  # 转成字符串
                "entity_set": meta.get("Entity_set", []),
                "evidence": meta.get("Evidence", {}),
                "types": meta.get("types", []),
            })
        
        dev_data = None  # factkg 没有 dev
        return dev_data, test_data
    elif data == "truthfulqa":
        file_path = "/mnt/petrelfs/linzhouhan/weirubin/projects/neuralKNN/downstream/data/TruthfulQA/TruthfulQA.csv"
        df = pd.read_csv(file_path)
        test_data = []
        for i, row in df.iterrows():
            test_data.append({
                "id": f"truthfulqa-{i}",
                "question": row["Question"],
                "answer": [a.strip() for a in str(row["Correct Answers"]).split(";") if a.strip()],
                "category": row["Category"],
                "type": row["Type"],
                "best_answer": row["Best Answer"],
                "best_incorrect": row["Best Incorrect Answer"],
                "incorrect_answers": row["Incorrect Answers"],
            })
        dev_data = None
        return dev_data, test_data
    else:
        # 其他任务保持不变
        test_path = f"/mnt/petrelfs/linzhouhan/weirubin/projects/neuralKNN/downstream/data/{data}/test.jsonl"
        test_data = None
        if os.path.isfile(test_path):
            test_data = get_jsonl(test_path)
        dev_data = None
        return dev_data, test_data

if __name__ == "__main__":

    args = parse_args()

    ## load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side = 'left',
        add_eos_token=False, ## import to include this!
        use_fast=False,
    )
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    ## load retriever and retriever_tokenizer
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    retrieval_embed_length = 0

    ## prepare prompt
    dev_data,test_data = load_dataset(
        args.data,
        args,
    )

    if args.max_test_samples is not None:
        test_data = test_data[:args.max_test_samples]
    
    if args.use_rag:
        rag_searcher = load_contrieverFaiss(args.top_k, args.nprobe, args.index_dir)
    else:
        rag_searcher = None

    t0 = time.time()
    prompts,backgrounds = prepare_prompts(
        dev_data = dev_data,
        test_data = test_data,
        task_type = args.task_type,
        tokenizer = tokenizer,
        n_shot = args.n_shot,
        use_rag = args.use_rag,
        retrieval_embed_length = retrieval_embed_length,
        chat_format = args.chat_format, 
        rag_searcher = rag_searcher,
        model_type = args.model_type,
        args=args
    )
    t1 = time.time()

    os.makedirs(args.results_dir, exist_ok=True)
    first_prompt_path = os.path.join(args.results_dir, "first_prompt.txt")
    with open(first_prompt_path, 'w', encoding='utf-8') as f:
        f.write(prompts[0])
    print(f"✅ First prompt saved to: {first_prompt_path}")
    
    retrieval_embeds = None

    avg_prompt_length = tokenizer(prompts,return_length=True).length
    avg_prompt_length = sum(avg_prompt_length)/len(avg_prompt_length)
    

    ## load llm
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    MODEL_CLASS = eval(config.architectures[0])
    model = MODEL_CLASS.from_pretrained(
        args.model_name_or_path,
        torch_dtype = torch.bfloat16,
        low_cpu_mem_usage = True,
        device_map='auto',
    )
    model.eval()
    # model = model.to(device)

    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    vocab_size = model.config.vocab_size

    if args.task_type in ['open_qa','fact_checking']:
        generated_results, profile_info = llm_for_open_generation(
            llm = model,
            llm_tokenizer = tokenizer,
            prompts = prompts,
            retrieval_embeds = retrieval_embeds,
            batch_size = args.eval_batch_size,
            enable_progress_bar= args.enable_progress_bar,
        )
    t2 = time.time()

    answers = [x['answer'] for x in test_data]
    if args.eval_metrics == 'substring_match':
        score,score_per_sample = get_substring_match_score(generated_results,answers)
    elif args.eval_metrics == 'fact_checking_acc':
        score,score_per_sample = eval_fact_checking(generated_results,answers)
    elif args.eval_metrics == 'truthfulqa_f1_rl':
        f1,rl,f1_scores,rl_scores = eval_truthfulqa(generated_results,answers)
        score = f"{f1}-{rl}"
        score_per_sample = [(f1_score,rl_score) for f1_score,rl_score in zip(f1_scores,rl_scores)]

    result_dict =   {
        "dataset":args.data,
        "batch_size":args.eval_batch_size,
        "include_retrieval":args.use_rag,
        "avg_prompt_length":avg_prompt_length,
        "model":args.model_name_or_path,
        f"{args.eval_metrics}":score,
    }

    os.makedirs(args.results_dir, exist_ok=True)
    file_path = os.path.join(args.results_dir, "results.json")

    # 保存为 JSON 文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    print(f"Results saved to {file_path}")

    print(json.dumps(result_dict,indent=4))

    # ========= profiling =========
    avg_cuda_time = sum(profile_info["cuda_time_per_batch"]) / len(profile_info["cuda_time_per_batch"])
    total_tokens = sum(profile_info["tokens_per_batch"])

    flops_per_token = estimate_flops_per_token(model, hidden_size, num_layers, vocab_size, seq_len=avg_prompt_length)
    gflops_per_token = flops_per_token / 1e9

    profile_dict = {
        "prepare_prompts_time_s": t1 - t0,
        "generation_time_s": t2 - t1,
        "total_time_s": t2 - t0,
        "avg_cuda_time_per_batch_ms": avg_cuda_time,
        "gflops_per_token": gflops_per_token,
        "total_generated_tokens": total_tokens,
        "total_tokens": profile_info["total_tokens"]
    }

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "profile.json"), "w") as f:
        json.dump(profile_dict, f, indent=4)

    print("Profiling saved to profile.json")
