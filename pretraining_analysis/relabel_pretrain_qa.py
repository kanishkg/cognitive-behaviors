import datasets
import os
import math
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse
import re


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='openwebmath_none', help='Dataset name')
parser.add_argument('--start', type=int, default=-1, help='Shard to process')
parser.add_argument('--end', type=int, default=-1, help='Number of shards to process')
parser.add_argument('--split', type=str, default='train', help='Split to process')
parser.add_argument('--max_examples', type=int, default=-1, help='Max examples to process')
parser.add_argument('--save_every', type=int, default=10000, help='Save every N examples')
parser.add_argument('--user', type=str, default='', help='User to push the dataset to')
parser.add_argument('--method', type=str, default='curated', help='Method to use, curated or negative')

def get_prompts(ds, tokenizer, prompt_templates, method):
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
        if method == 'curated':
            prompt = prompt_templates['qa'] + f"\n<text>\n{example}\n</text>"
        elif method == 'negative':
            prompt = prompt_templates['qa_none'] + f"\n<text>\n{example}\n</text>"
        else:
            raise ValueError(f"Unknown method: {method}")
        prompt = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': ''}]
        prompts += [prompt]
  
    new_prompts = [tokenizer.apply_chat_template(
        p,
        tokenize=False,
    ) for p in prompts]
    return new_prompts

def parse_output(output):
    query_match = re.search(r'<question>(.*?)</question>', output, re.DOTALL)
    think_match = re.search(r'<thoughts>(.*?)</thoughts>', output, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL)
    
    query = query_match.group(1) if query_match else ""
    think = think_match.group(1) if think_match else ""
    answer = answer_match.group(1) if answer_match else ""
    
    completion = f"<think>{think}</think>\n<answer>{answer}</answer>" if think or answer else ""
    return query, completion

def main(args):
    prompt_templates = {
    'qa': """Your goal is to split the text into a question, thought and an answer.

Make sure that the question is in the text. 
Make sure that the answer and the process of the writer to get to the answer are in the text.
Paraphrase the answer so that the answer is cleaned up. Make sure that the answer has the process of finding the solution.
Like backtracking, or verifying the answer, or setting subgoals.
Here are the definitions of the words:
Backtracking: The process of finding the solution by going back and forth between the answer and the question.
Verification: The process of checking the answer to see if it is correct.
Subgoal setting: The process of setting smaller goals to reach the final answer.

Write the question in <question>...</question>.
For the answer, split the answer into the process towards reaching the answer and the final answer.
Write the process in <thoughts>thinking process of the author with backtracking etc.</thoughts> and the final answer in <answer>...</answer>.
Use first person pronouns like "I" and "me" to refer to the author.
So, the thoughts should be in the first person, and should look like the author is thinking out loud. Eg: "I think I should try this next."
Include the mistakes made by the author in the thoughts section. If the author makes a mistake, include the mistake in the thoughts section.
Use present tense in the thoughts section. The thoughts section should look like the author is thinking out loud.
This will come with the realization from the author that they made a mistake.

Now do it for this text:""",

    'qa_none': """Your goal is to split the text into a question, thought and an answer.
Make sure that the question, thoughts and answer are in the text. 
Paraphrase the answer so that the answer is cleaned up. Make sure that the answer has steps to find the solution.    
Write the question in <question>...</question>.
Write the process in <thoughts>steps to find the solution</thoughts> and the final answer in <answer>...</answer>.

Now do it for this text:""",
}

    if args.dataset_name == 'openwebmath_backtrack':
        ds = datasets.load_dataset(f'{args.user}/open-web-math-backtrack-processed', num_proc=os.cpu_count()-2, split=args.split)
    elif args.dataset_name == 'openwebmath_none':
        ds = datasets.load_dataset(f'{args.user}/open-web-math-none-processed', num_proc=os.cpu_count()-2, split=args.split)
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")
        
    if args.max_examples > 0:
        ds = ds.select(range(args.max_examples))
    
    if args.start >= 0 and args.end >= 0 and args.start < args.end:
        print('Subsampling the dataset with start={} and end={}'.format(args.start, args.end))
        ds = ds.select(range(args.start, args.end))
    
    # filter examples where 'contain_problem' is no or 'contain_solution' is no
    if args.dataset_name == 'openwebmath_backtrack' or args.dataset_name == 'openwebmath_none':
        ds = ds.filter(lambda x: x['contain_problem'] != 'no' and x['contain_solution'] != 'no')
        print(f"Number of examples after filtering: {len(ds)}")

    llm = LLM(
        model='Qwen/Qwen2.5-32B-Instruct',
        tokenizer_mode="auto",
        max_num_seqs=64,
        enable_prefix_caching=True,
        trust_remote_code=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95,
        max_model_len=8192,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")

    num_batches = math.ceil(len(ds) / args.save_every)
    all_ds = []
    for shard_idx in tqdm(range(num_batches), desc='Shards'):
        batch_start = shard_idx * args.save_every
        batch_end = min((shard_idx + 1) * args.save_every, len(ds))
        
        curr_batch = ds.select(range(batch_start, batch_end))
        prompts = get_prompts(curr_batch, tokenizer, prompt_templates, args.method)
        sampling_params = SamplingParams(
            max_tokens=4096+1024,
            temperature=0,
        )

        responses = llm.generate(prompts, sampling_params=sampling_params)

        outputs_dict = {
            'raw_qa': [None] * len(curr_batch),
            'query': [None] * len(curr_batch),
            'completion': [None] * len(curr_batch)
        }
        
        for i, response in enumerate(responses):
            output = response.outputs[0].text.strip()
            query, completion = parse_output(output)
            outputs_dict['raw_qa'][i] = output
            outputs_dict['query'][i] = query
            outputs_dict['completion'][i] = completion
        
        curr_batch = curr_batch.add_column('raw_qa', outputs_dict['raw_qa'])
        curr_batch = curr_batch.add_column('query', outputs_dict['query'])
        curr_batch = curr_batch.add_column('completion', outputs_dict['completion'])

        all_ds.append(curr_batch)
        
        # Save the dataset
        try:
            ds_so_far = datasets.concatenate_datasets(all_ds)
            if args.start >= 0 and args.end >= 0 and args.start < args.end:
                suffix = f'_{args.start}_{args.end}'
            else:
                suffix = ''
            ds_out_name = f'{args.user}_{args.dataset_name}_{suffix}'
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
        ds_out_name = f'{args.user}_{args.dataset_name}_{suffix}'
        ds_so_far.push_to_hub(ds_out_name)
    except Exception as e:
        print(f'Final error saving dataset: {e}')
    print('Done')
    
if __name__ == '__main__':

    args = parser.parse_args()
    main(args)
