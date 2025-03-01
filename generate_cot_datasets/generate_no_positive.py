import argparse
import os
import json
import torch
import re
import gc
from vllm import LLM, SamplingParams
from verl.utils.reward_score.countdown import compute_score
from tqdm import tqdm

from tasks.countdown import CountDown

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

def main():
    parser = argparse.ArgumentParser(
        description="Generate responses using vLLM on an HF parquet dataset and score them."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="responses.jsonl"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1024
    )

    args = parser.parse_args()
    countdown = CountDown(
            start_probs=[0., 0.5, 0.5],
            # start_probs=[0.0, 0.0, 1.],
            max_target=100,
            min_target=10
        )

    dataset = [countdown.get_task(return_raw=True) for _ in range(4 * args.num_samples)]

    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=args.temperature,
    )

    print(f"Using checkpoint {args.ckpt}")
    llm = LLM(
        model=args.ckpt,
        enable_prefix_caching=False,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,
    )

    prompts, ground_truths = [], []
    for example in tqdm(dataset, desc="Generating responses"):
        prompt = example["query"]
        target = re.search(r'equals (\d+)', prompt).group(1)
        numbers_pattern = r'the numbers\s*\[([^\]]+)\]'
        match_nums = re.search(numbers_pattern, prompt)
        if match_nums:
            str_nums = match_nums.group(1)
            nums_list = [int(s.strip()) for s in str_nums.split(',')]
        else:
            nums_list = []

        ground_truth = {
            'target': int(target),
            'numbers': nums_list
        }

        prompts.append(prompt)
        ground_truths.append(ground_truth)

    responses = llm.generate(prompts, sampling_params=sampling_params)

    results = []
    for index, (prompt, ground_truth, response) in enumerate(zip(prompts, ground_truths, responses)):
        generated_text = response.outputs[0].text.strip()
        generated_text = f"Assistant:\n{generated_text}"
        score = compute_score(generated_text, ground_truth, format_score=0., score=1.)

        if not bool(score):
            result_record = {
                "query": prompt,
                "completion": generated_text,
            }
            results.append(result_record)
        
            if len(results) >= args.num_samples:
                break

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved responses and scores to {args.output_path}")

    del llm
    gc.collect()  # Collect garbage
    torch.cuda.empty_cache()  # Clear CUDA cache
    torch.distributed.destroy_process_group()  # Destroy the process group

if __name__ == "__main__":
    main()