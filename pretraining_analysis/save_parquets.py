import os
import argparse

import datasets
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right'

max_tokens = 4090
margin_tokens = 0

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='', help='Dataset name')
args = parser.parse_args()

dataset_name = args.dataset_name
assert dataset_name is not None, "Dataset name is not set"

ds = datasets.load_dataset(f'{dataset_name}')
if 'test' not in ds:
    ds = ds['train'].train_test_split(test_size=0.1, seed=42)

def filter_fn(example):
    curr_query = example['query']
    curr_completion = example['completion']
    
    if not curr_query or not curr_completion: # skip empty queries or completions
        return False
    
    query_tok = tokenizer(curr_query, truncation=True, max_length=max_tokens-margin_tokens)
    query_len = len(query_tok['input_ids'])
    
    if query_len > max_tokens - margin_tokens:
        return False
    return True
ds = ds.filter(filter_fn, num_proc=os.cpu_count())

def map_fn(example):
    curr_query = example['query']
    curr_completion = example['completion']
    query_tok = tokenizer(curr_query, truncation=True, max_length=max_tokens-margin_tokens)
    completion_tok = tokenizer(curr_completion, truncation=True, max_length=max_tokens-margin_tokens)
    total_len = len(query_tok['input_ids']) + len(completion_tok['input_ids'])
    
    if total_len > max_tokens:
        len_query = len(query_tok['input_ids'])
        len_completion = len(completion_tok['input_ids'])
        if len_query > max_tokens - margin_tokens:
            len_query = max_tokens - margin_tokens
            len_completion = 0
        else:
            len_completion = max_tokens - margin_tokens - len_query
        query_tok['input_ids'] = query_tok['input_ids'][:len_query]
        query_tok['attention_mask'] = query_tok['attention_mask'][:len_query]
        completion_tok['input_ids'] = completion_tok['input_ids'][:len_completion]
        completion_tok['attention_mask'] = completion_tok['attention_mask'][:len_completion]
        
        new_query = tokenizer.decode(query_tok['input_ids'])
        new_completion = tokenizer.decode(completion_tok['input_ids'])
        
        example['query'] = new_query
        example['completion'] = new_completion
    
    return example
    

ds = ds.map(map_fn, num_proc=os.cpu_count())
ds = ds.filter(filter_fn, num_proc=os.cpu_count())
train_completion = ds['train']['completion']
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
tokens = tokenizer(train_completion)
lens = [len(t) for t in tokens['input_ids']]
print(f"Max length: {max(lens)}")
print(f"Min length: {min(lens)}")
print(f"Mean length: {np.mean(lens)}")
print(f"Median length: {np.median(lens)}")
print(f"Total tokens: {sum(lens)}")
print(f"Number of completions: {len(lens)}")

ds.push_to_hub(dataset_name)

os.system(f'mkdir -p {dataset_name}/')
ds['train'].to_parquet(f'{dataset_name}/train.parquet')
ds['test'].to_parquet(f'{dataset_name}/test.parquet')