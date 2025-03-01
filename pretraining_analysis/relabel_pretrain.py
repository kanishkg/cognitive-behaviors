import datasets
import os
import math
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=-1, help='Shard to process')
parser.add_argument('--end', type=int, default=-1, help='Number of shards to process')
parser.add_argument('--split', type=str, default='train', help='Split to process')
parser.add_argument('--max_examples', type=int, default=-1, help='Max examples to process')
parser.add_argument('--save_every', type=int, default=10000, help='Save every N examples')
parser.add_argument('--user', type=str, default='', help='User to push the dataset to')
parser.add_argument('--dataset_name', type=str, default='open-web-math', help='Dataset to process')

PROMPT_LOC_DICT = {
    'backtracking': './pretraining_analysis/prompts/backtracking_v0.txt',
    'verification': './pretraining_analysis/prompts/verification_v0.txt',
    'subgoal_setting': './pretraining_analysis/prompts/subgoal_setting_v0.txt',
    'backward_chaining': './pretraining_analysis/prompts/backward_chaining_v0.txt',
}


def get_prompts(ds, tokenizer, prompt_templates):
    prompts = []
    tokenized_inputs = tokenizer(ds['text'])
    samples = []
    max_seq_length = 4096
    for e, example in tqdm(enumerate(tokenized_inputs['input_ids']), desc="Truncating prompts"):
        if len(example) > max_seq_length-1024:
            sample = tokenizer.decode(example[: max_seq_length - 1024])
            sample = sample[: sample.rfind("\n")]
            samples += [sample]
        else:
            samples += [ds['text'][e]]

    for example in tqdm(samples, desc="Generating prompts"):
        if args.only_subgoal:
            subgoal_setting_prompt = prompt_templates['subgoal_setting'].format(response=example)
            subgoal_setting_prompt = [{'role': 'user', 'content': subgoal_setting_prompt}]
            prompts += [subgoal_setting_prompt]
            continue
        backtracking_prompt = prompt_templates['backtracking'].format(response=example)
        backtracking_prompt = [{'role': 'user', 'content': backtracking_prompt}]
        verification_prompt = prompt_templates['verification'].format(response=example)
        verification_prompt = [{'role': 'user', 'content': verification_prompt}]
        subgoal_setting_prompt = prompt_templates['subgoal_setting'].format(response=example)
        subgoal_setting_prompt = [{'role': 'user', 'content': subgoal_setting_prompt}]
        backward_chaining_prompt = prompt_templates['backward_chaining'].format(response=example)
        backward_chaining_prompt = [{'role': 'user', 'content': backward_chaining_prompt}]
        prompts += [backtracking_prompt, verification_prompt, subgoal_setting_prompt, backward_chaining_prompt]
    new_prompts = [tokenizer.apply_chat_template(
        p,
        tokenize=False,
    ) for p in prompts]

    return new_prompts

def main(args):
    prompt_templates = {
            k: open(v).read() for k, v in PROMPT_LOC_DICT.items()
        }
    for k, v in prompt_templates.items():
        assert '{response}' in v, f'Prompt {k} does not contain {{response}} in {v}'

    if args.dataset_name == 'open-web-math':
        ds = datasets.load_dataset('open-web-math/open-web-math', num_proc=os.cpu_count()-2, split=args.split)
    elif args.dataset_name == 'finemath':
        ds = datasets.load_dataset('HuggingFaceTB/finemath', 'finemath-4plus', num_proc=os.cpu_count()-2, split=args.split)
    else:
        raise ValueError(f'Unknown dataset: {args.dataset_name}')

    if args.max_examples > 0:
        ds = ds.select(range(args.max_examples))
    
    if args.start >= 0 and args.end >= 0 and args.start < args.end:
        print('Subsampling the dataset with start={} and end={}'.format(args.start, args.end))
        ds = ds.select(range(args.start, args.end))

    llm = LLM(
        model='Qwen/Qwen2.5-32B-Instruct',
        tokenizer_mode="auto",
        max_num_seqs=32,
        enable_prefix_caching=True,
        trust_remote_code=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95,
        max_model_len=4096,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")

    num_batches = math.ceil(len(ds) / args.save_every)
    batch_size = args.save_every
    all_ds = []
    for shard_idx in tqdm(range(num_batches), desc='Shards'):
        batch_start = shard_idx * args.save_every
        batch_end = min((shard_idx + 1) * args.save_every, len(ds))
        
        curr_batch = ds.select(range(batch_start, batch_end))
        prompts = get_prompts(curr_batch, tokenizer, prompt_templates)

        sampling_params = SamplingParams(
            max_tokens=1024,
            temperature=0,
        )
        responses = llm.generate(prompts, sampling_params=sampling_params)

        outputs_dict = {
            'backtracking_raw': [None] * len(curr_batch),
            'verification_raw': [None] * len(curr_batch),
            'subgoal_setting_raw': [None] * len(curr_batch),
            'backward_chaining_raw': [None] * len(curr_batch)
        }
        
        for i, response in enumerate(responses):
            output = response.outputs[0].text.strip()
            if args.only_subgoal:
                outputs_dict['subgoal_setting_raw'][i] = output
                continue
            idx = i % 4
            batch_idx = i // 4
            if idx == 0:
                outputs_dict['backtracking_raw'][batch_idx] = output
            elif idx == 1:
                outputs_dict['verification_raw'][batch_idx] = output
            elif idx == 2:
                outputs_dict['subgoal_setting_raw'][batch_idx] = output
            elif idx == 3:
                outputs_dict['backward_chaining_raw'][batch_idx] = output

        curr_batch = curr_batch.add_column('backtracking_raw', outputs_dict['backtracking_raw'])
        curr_batch = curr_batch.add_column('verification_raw', outputs_dict['verification_raw'])
        curr_batch = curr_batch.add_column('subgoal_setting_raw', outputs_dict['subgoal_setting_raw'])
        curr_batch = curr_batch.add_column('backward_chaining_raw', outputs_dict['backward_chaining_raw'])

        all_ds.append(curr_batch)
        
        # Save the dataset
        try:
            ds_so_far = datasets.concatenate_datasets(all_ds)
            if args.start >= 0 and args.end >= 0 and args.start < args.end:
                suffix = f'_{args.start}_{args.end}'
            else:
                suffix = ''
            ds_out_name = f'{args.user}{args.dataset_name}_raw_v3{suffix}'
            ds_so_far.push_to_hub(ds_out_name)
        except Exception as e:
            print(f'Error saving dataset: {e}')
            continue
    
    try:
        ds_so_far = datasets.concatenate_datasets(all_ds)
        if args.start >= 0 and args.end >= 0 and args.start < args.end:
            suffix = f'_{args.start}_{args.end}'
        else:
            suffix = ''
        ds_out_name = f'{args.user}{args.dataset_name}_raw_v3{suffix}'
        ds_so_far.push_to_hub(ds_out_name)
    except Exception as e:
        print(f'Final error saving dataset: {e}')
    print('Done')
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
