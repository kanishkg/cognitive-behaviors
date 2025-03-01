import argparse
import os
import json
import torch
import re
import gc
from vllm import LLM, SamplingParams
from datasets import load_dataset
from verl.utils.reward_score.countdown import compute_score as compute_score_countdown
from tqdm import tqdm

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
        "--dataset",
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

    parser.add_argument(
        "--task-type",
        type=str,
        options=["countdown"],
        required=True,
        help="The type of task the model was trained on."
    )


    args = parser.parse_args()
    ckpt_dir = args.ckpt
    dataset_path = args.dataset
    dataset = load_dataset("parquet", data_files=dataset_path)['train']

    sampling_params = SamplingParams(
        max_tokens=1024 if args.task_type == "countdown" else 2048,
        temperature=args.temperature,
    )

    if 'global_step' not in ckpt_dir:
        ckpts = os.listdir(ckpt_dir)
        steps = [re.search(r'global_step_(\d+)', ckpt).group(1) for ckpt in ckpts]
        ckpts = sorted([(int(step), ckpt) for step, ckpt in zip(steps, ckpts) if int(step) > 40], key=lambda x: x[0])
    else:
        step = re.search(r'global_step_(\d+)', ckpt_dir).group(1)
        if step is None:
            step = 0
        ckpts = [(int(step), ckpt_dir)]


    for step, ckpt in ckpts:
        model_ckpt = os.path.join(ckpt_dir, ckpt)
        print(f"Using checkpoint {model_ckpt}")
        llm = LLM(
            model=model_ckpt,
            enable_prefix_caching=False,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
        )

        prompts, ground_truths = [], []
        for example in tqdm(dataset, desc="Generating responses"):
            prompt = example["prompt"][0]["content"]
            ground_truth = example["reward_model"]["ground_truth"]
            extra_info = example.get("extra_info", {})
            index = extra_info.get("index", None)
            data_source = example.get("data_source", "")

            prompts.append(prompt)
            ground_truths.append(ground_truth)

        responses = llm.generate(prompts, sampling_params=sampling_params)

        results = []
        for index, (prompt, ground_truth, response) in enumerate(zip(prompts, ground_truths, responses)):
            generated_text = response.outputs[0].text.strip()
            if args.task_type == "countdown":
                generated_text = f"Assistant:\n{generated_text}"
                score = compute_score_countdown(generated_text, ground_truth, format_score=0., score=1.)
            else:
                raise ValueError(f"Unknown task type {args.task_type}")

            result_record = {
                "index": index,
                "data_source": data_source,
                "prompt": prompt,
                "generated": generated_text,
                "score": score,
                "ground_truth": ground_truth,
            }
            results.append(result_record)

        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path, exist_ok=True)

        output_path = os.path.join(args.output_path, f"completions_step{step}.jsonl")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved responses and scores to {output_path}")

        del llm
        gc.collect()  # Collect garbage
        torch.cuda.empty_cache()  # Clear CUDA cache
        torch.distributed.destroy_process_group()  # Destroy the process group


if __name__ == "__main__":
    main()