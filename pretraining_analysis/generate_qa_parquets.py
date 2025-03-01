import os
import numpy as np
import datasets
from transformers import AutoTokenizer
from tqdm import tqdm

# add hf dataset names here
data_names = []
all_ds = []
for data_name in data_names:
    ds = datasets.load_dataset(data_name)
    # use train split
    ds = ds['train']
    all_ds.append(ds)


ds = datasets.concatenate_datasets(all_ds)

# filter out empty completions and queries
print(f"Number of examples: {len(ds)}")
ds = ds.filter(lambda x: len(x['query']) > 0 and len(x['completion']) > 0)
print(f"Number of examples: {len(ds)}")

prefix = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: {query} Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags.
Assistant: Let me solve this step by step."""

# add prefix to query
ds = ds.map(lambda x: {'query': prefix.format(query=x['query']), 'completion': '\n'+x['completion']})

# delete all columns except query and completion
ds = ds.remove_columns([col for col in ds.column_names if col not in ['query', 'completion']])


train_completion = ds['completion']
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
tokens = tokenizer(train_completion)
lens = [len(t) for t in tokens['input_ids']]
print(f"Max length: {max(lens)}")
print(f"Min length: {min(lens)}")
print(f"Mean length: {np.mean(lens)}")
print(f"Median length: {np.median(lens)}")
print(f"Total tokens: {sum(lens)}")
print(f"Number of completions: {len(lens)}")
# do the same for queries
query_tokens = tokenizer(ds['query'])
query_lens = [len(t) for t in query_tokens['input_ids']]
print(f"Max query length: {max(query_lens)}")
print(f"Min query length: {min(query_lens)}")
print(f"Mean query length: {np.mean(query_lens)}")
print(f"Median query length: {np.median(query_lens)}")
print(f"Total query tokens: {sum(query_lens)}")
print(f"Number of queries: {len(query_lens)}")

target_len = 83000000
cumsum = 0
keep_idx = []
for i, l in enumerate(lens):
    # clip l at 4096
    l = min(l, 4096)
    if cumsum + l <= target_len:
        cumsum += l
        keep_idx.append(i)
    else:
        break

ds = ds.select(keep_idx)
print(f"Kept {len(keep_idx)} examples with total {cumsum} tokens")

train_completion = ds['completion']
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
tokens = tokenizer(train_completion)
lens = [len(t) for t in tokens['input_ids']]
print(f"Max length: {max(lens)}")
print(f"Min length: {min(lens)}")
print(f"Mean length: {np.mean(lens)}")
print(f"Median length: {np.median(lens)}")
print(f"Total tokens: {sum(lens)}")
print(f"Number of completions: {len(lens)}")

ds_out_name = 'obiwan96/owm_nonev4'
ds = ds.train_test_split(test_size=0.05)
ds.push_to_hub(ds_out_name)

# save as train.parquet and test.parquet
if not os.path.exists('/home/kanishk/ba/data/owm_mathv4_none'):
    os.makedirs('/home/kanishk/ba/data/owm_mathv4_none')
ds['train'].to_parquet('/home/kanishk/ba/data/owm_mathv4_none/train.parquet')
ds['test'].to_parquet('/home/kanishk/ba/data/owm_mathv4_none/test.parquet')