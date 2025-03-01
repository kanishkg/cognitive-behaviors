import json
import tempfile
from openai import OpenAI
import asyncio
import argparse
import os
import random
import re
import fcntl

class BaseAnalyzer:
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key)
    
    def create_analysis_prompts(self, numbers, target, completion):
        """Create the analysis prompts for a given problem."""
        pass

    def convert_to_batch_format(self, input_path, num_samples=200):
        """
        Convert arithmetic problems to OpenAI Batch API format with all analysis prompts.
        If input_path is a directory, combine all JSON files whose names match
        the pattern 'completion_step<d+>' (e.g. completion_step3.json). Each item is
        augmented with a 'step' field based on its filename.
        """
        tmp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        output_file = tmp_file.name
        data = []
        if os.path.isdir(input_path):
            print(f"Input path '{input_path}' is a directory. Processing all completion files matching pattern...")
            for fname in os.listdir(input_path):
                if re.search(r'completions_step(\d+)', fname):
                    full_path = os.path.join(input_path, fname)
                    try:
                        with open(full_path, 'r') as f:
                            file_data = json.load(f)
                            step_match = re.search(r'completions_step(\d+)', fname)
                            step = step_match.group(1) if step_match else '1'
                            if len(file_data) > num_samples:
                                file_data = random.sample(file_data, num_samples)
                            for item in file_data:
                                item['step'] = step
                                data.append(item)
                    except Exception as e:
                        print(f"Error reading file {full_path}: {e}")
        else:
            # Assume input_path is a single file.
            with open(input_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    if 'step' not in item:
                        item['step'] = '1'
        
        
        # Process each problem and write a batch request for each prompt.
        global_idx = 0
        with open(output_file, 'w') as f:
            for item in data:
                numbers = item['ground_truth']['numbers']
                target = item['ground_truth']['target']
                completion = item['generated']
                step = item.get('step', '1')
                
                prompts = self.create_analysis_prompts(numbers, target, completion)
                prompt_types = ['verification', 'backtracking', 'subgoal', 'backward-chaining']
                
                for prompt, prompt_type in zip(prompts, prompt_types):
                    # Include the step number in the custom_id.
                    custom_id = f"prob-{global_idx}-{step}-{prompt_type}"
                    batch_request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-4o-mini",
                            "messages": [
                                {"role": "system", "content": "You are a helpful assistant that analyzes mathematical reasoning."},
                                {"role": "user", "content": prompt}
                            ],
                            "max_tokens": 512,
                        }
                    }
                    f.write(json.dumps(batch_request) + '\n')
                global_idx += 1
        return output_file

    def upload_file(self, file_path):
        """Upload the JSONL file to OpenAI."""
        print("Uploading file...")
        with open(file_path, "rb") as file:
            file_obj = self.client.files.create(
                file=file,
                purpose="batch"
            )
        print(f"File uploaded with ID: {file_obj.id}")
        return file_obj.id

    def submit_batch(self, file_id):
        """Submit a batch job using the uploaded file."""
        print("Submitting batch job...")
        batch = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "Countdown reasoning analysis"
            }
        )
        print(f"Batch submitted with ID: {batch.id}")
        return batch.id

    def check_status(self, batch_id):
        """Check the status of a batch job."""
        batch = self.client.batches.retrieve(batch_id)
        return batch

    def download_results(self, file_id, output_path):
        """Download and save the results."""
        print(f"Downloading results to {output_path}...")
        response = self.client.files.content(file_id)
        with open(output_path, 'w') as f:
            f.write(response.text)
        print("Results downloaded successfully")
        return output_path

    async def poll_until_complete(self, batch_id, interval_seconds=60):
        """Poll the batch job status until it completes."""
        number_of_checks = 0
        while True:
            batch = self.check_status(batch_id)
            status = batch.status
            print(f"\nCurrent status: {status}")
            print(f"Progress: {batch.request_counts.completed}/{batch.request_counts.total} requests completed")
            if status == "completed":
                print("\nBatch job completed successfully!")
                return batch
            elif status in ["failed", "expired", "cancelled", "cancelling"]:
                print(f"\nBatch job ended with status: {status}")
                if batch.error_file_id:
                    print("Downloading error file...")
                    self.download_results(batch.error_file_id, "batch_errors.jsonl")
                return batch
            if number_of_checks > 2:
                interval_seconds = min(interval_seconds * 1.5, 300) if number_of_checks < 10 else min(interval_seconds * 2, 600)
            print(f"Waiting {interval_seconds} seconds before next check...")
            await asyncio.sleep(interval_seconds)
            number_of_checks += 1

    def process_results(self, batch_input_file, results_file, output_dir):
        """
        Process the batch results (from a single combined batch file)
        and generate detailed evaluation files grouped by step as well as
        an updated aggregate results file.
        """
        print("\nProcessing results...")
        # Build a lookup from the batch input file.
        input_data = [json.loads(line) for line in open(batch_input_file, 'r')]
        input_lookup = {item['custom_id']: item for item in input_data}
        
        with open(results_file, 'r') as f:
            results = [json.loads(line) for line in f]
        
        # Group results by step and then by problem id.
        results_by_step = {}
        for result in results:
            custom_id = result['custom_id']
            parts = custom_id.split('-')
            if len(parts) < 4:
                print(f"Unexpected custom_id format: {custom_id}")
                continue
            # parts[0] is "prob", parts[1] is global problem id, parts[2] is step, parts[3] is prompt type.
            prob_id = parts[1]
            step = parts[2]
            prompt_type = parts[3]
            # Normalize the key for backward-chaining responses.
            key_prefix = "backward" if prompt_type == "backward-chaining" else prompt_type

            if step not in results_by_step:
                results_by_step[step] = {}
            if prob_id not in results_by_step[step]:
                results_by_step[step][prob_id] = {}
            try:
                response_text = result['response']['body']['choices'][0]['message']['content']
                count_match = re.search(r'<count>(\d+)</count>', response_text)
                count = int(count_match.group(1)) if count_match else 0

                results_by_step[step][prob_id][f"{key_prefix}_count"] = count
                results_by_step[step][prob_id][f"{key_prefix}_response"] = response_text
                if custom_id in input_lookup:
                    results_by_step[step][prob_id][f"{key_prefix}_query"] = input_lookup[custom_id]['body']['messages'][1]['content']
                else:
                    results_by_step[step][prob_id][f"{key_prefix}_query"] = ""
            except (KeyError, IndexError) as e:
                print(f"Error processing result {custom_id}: {str(e)}")
                continue
        
        # For each step, compute aggregated metrics and write detailed output.
        aggregate_results = {}
        for step, problems in results_by_step.items():
            num_problems = len(problems)
            metrics = {
                'verification_count': sum(p.get('verification_count', 0) for p in problems.values()),
                'backtracking_count': sum(p.get('backtracking_count', 0) for p in problems.values()),
                'subgoal_count': sum(p.get('subgoal_count', 0) for p in problems.values()),
                'backward_count': sum(p.get('backward_count', 0) for p in problems.values())
            }
            avg_metrics = {
                'avg_verifications': metrics['verification_count'] / num_problems if num_problems else 0,
                'avg_backtracking': metrics['backtracking_count'] / num_problems if num_problems else 0,
                'avg_subgoals': metrics['subgoal_count'] / num_problems if num_problems else 0,
                'avg_backwards': metrics['backward_count'] / num_problems if num_problems else 0
            }
            step_results = {
                'avg_verifications': avg_metrics['avg_verifications'],
                'avg_backtracking': avg_metrics['avg_backtracking'],
                'avg_subgoals': avg_metrics['avg_subgoals'],
                'avg_backwards': avg_metrics['avg_backwards'],
                'total_verifications': metrics['verification_count'],
                'total_backtracking': metrics['backtracking_count'],
                'total_subgoals': metrics['subgoal_count'],
                'total_backwards': metrics['backward_count']
            }
            aggregate_results[step] = step_results

            # Save detailed results for this step.
            os.makedirs(output_dir, exist_ok=True)
            detailed_path = os.path.join(output_dir, f"evaluation_step{step}.json")
            with open(detailed_path, 'w') as f:
                json.dump({
                    'results': list(problems.values()),
                    'metrics': {**metrics, **avg_metrics}
                }, f, indent=2)
            print(f"\nStep {step} complete:")
            print(f"  Average verifications: {avg_metrics['avg_verifications']:.4f}")
            print(f"  Average backtracking:  {avg_metrics['avg_backtracking']:.4f}")
            print(f"  Average subgoals:      {avg_metrics['avg_subgoals']:.4f}")
            print(f"  Average backwards:     {avg_metrics['avg_backwards']:.4f}")
            print(f"  Total verifications:   {metrics['verification_count']}")
            print(f"  Total backtracking:    {metrics['backtracking_count']}")
            print(f"  Total subgoals:        {metrics['subgoal_count']}")
            print(f"  Total backwards:       {metrics['backward_count']}")
            print(f"Detailed results saved to: {detailed_path}")
        
        # Update aggregate results file.
        aggregate_path = os.path.join(output_dir, "all_results.json")
        if os.path.exists(aggregate_path):
            with open(aggregate_path, 'r+') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    existing_results = json.load(f)
                    existing_results['results_by_step'].update(aggregate_results)
                    # Update steps_processed (sort numerically)
                    all_steps = set(existing_results.get('steps_processed', [])) | set(aggregate_results.keys())
                    existing_results['steps_processed'] = sorted(list(all_steps), key=lambda x: int(x))
                    f.seek(0)
                    f.truncate()
                    json.dump(existing_results, f, indent=2)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        else:
            final_aggregate = {
                'results_by_step': aggregate_results,
                'steps_processed': sorted(list(aggregate_results.keys()), key=lambda x: int(x))
            }
            with open(aggregate_path, 'w') as f:
                json.dump(final_aggregate, f, indent=2)
        print(f"\nAggregate results saved to: {aggregate_path}")
        os.remove(results_file)

    def process_synchronously(self, input_file, num_samples=200):
        """Process problems using synchronous API calls."""
        print("Processing problems synchronously...")
        with open(input_file, 'r') as f:
            data = json.load(f)
        if len(data) > num_samples:
            data = random.sample(data, num_samples)
        problem_metrics = {}
        for idx, item in enumerate(data):
            print(f"\rProcessing problem {idx + 1}/{len(data)}", end='')
            numbers = item['ground_truth']['numbers']
            target = item['ground_truth']['target']
            completion = item['generated']
            prompts = self.create_analysis_prompts(numbers, target, completion)
            prompt_types = ['verification', 'backtracking', 'subgoal', 'backward-chaining']
            problem_metrics[str(idx)] = {}
            for prompt, prompt_type in zip(prompts, prompt_types):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that analyzes mathematical reasoning."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=512,
                    )
                    response_text = response.choices[0].message.content
                    count_match = re.search(r'<count>(\d+)</count>', response_text)
                    count = int(count_match.group(1)) if count_match else 0
                    problem_metrics[str(idx)][f"{prompt_type}_count"] = count
                    problem_metrics[str(idx)][f"{prompt_type}_response"] = response_text
                    problem_metrics[str(idx)][f"{prompt_type}_query"] = prompt
                except Exception as e:
                    print(f"\nError processing {prompt_type} for problem {idx}: {str(e)}")
                    continue
        print("\nProcessing complete!")
        return problem_metrics


class CountdownAnalyzer(BaseAnalyzer):
    def __init__(self, api_key=None):
        super().__init__(api_key=api_key)

    def create_analysis_prompts(self, numbers, target, completion):
        prompts = [
            # 1. Answer-verification steps
            f"""Here is a chain-of-reasoning that a Language Model generated while trying to play the game of CountDown with the numbers {numbers}. The goal is to reach the target number {target}. The chain-of-reasoning the model used is: {completion}. 
Evaluate whether the chain-of-reasoning contains any answer-verification steps. An example of an answer-verification step is: 'This sequence results in 1, which is not equal to 22' and 'Since 25 is not equal to 22' for explicit verification and 'Too high!' or 'This works!' for implicit verification. We want to mark instances where the chain-of-reasoning explicitly checks the current result against the target number. 
If you find any answer-verification steps, please count them and provide the count as between the tags <count> </count>. If the chain-of-reasoning does not contain any answer-verification steps, please provide a count of 0 as <count>0</count>.""",

            # 2. Backtracking behavior
            f"""Here is a chain-of-reasoning that a Language Model generated while trying to play the game of CountDown with the numbers {numbers}. The goal is to reach the target number {target}. The chain-of-reasoning the model used is: {completion}.
Evaluate whether the chain-of-reasoning contains any backtracking behavior, where the model realizes a path won't work and explicitly goes back to try a different approach. Due to the nature of the problem, any attempt at a new combination of numbers that does not directly use the result from the previous computation is considered backtracking. 
For example, in the reasoning trace with numbers [20, 7, 11, 78] - "(78 - 20) - (11 - 7) = 58 - 4 = 54, (54 - 78) + 11 = -24 + 11 = -13, (-13 + 78) - 11 = 65 - 11 = 54, (78 - 58) + 11 = 20 + 11 = 31, (78 - 58) + (20 - 11) = 20 + 9 = 29, (78 - 20) + (11 - 7) = 58 + 4 = 62, (78 - 11) - (20 - 7) = 67 - 13 = 54, (78 - 20) + (11 / 7) = 58 + 1.5714 = 59.5714, (78 - 11) / (20 - 7) = 67 / 13 = 5, (78 - 20) + (11 + 7) = 58", there are 5 instances of backtracking to the initial numbers.
Count the number of distinct backtracking instances and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any backtracking behavior, please provide a count of 0 as <count>0</count>.""",

            # 3. Subgoal setting
            f"""Here is a chain-of-reasoning that a Language Model generated while trying to play the game of CountDown with the numbers {numbers}. The goal is to reach the target number {target}. The chain-of-reasoning the model used is: {completion}.
Evaluate whether the chain-of-reasoning contains any explicit subgoal setting, where the model breaks down the problem into smaller, intermediate goals. An example of subgoal setting is: "First, I'll try to get close to {target//2}, then...".
Count the number of distinct subgoals set and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any subgoal setting, please provide a count of 0 as <count>0</count>.""",

            # 4. Backward-chaining
            f"""Here is a chain-of-reasoning that a Language Model generated while trying to play the game of CountDown with the numbers {numbers}. The goal is to reach the target number {target}. The chain-of-reasoning the model used is: {completion}.
Evaluate whether the chain-of-reasoning contains any backward-chaining behavior, where the model starts from the target number and works backwards to the initial numbers. An example of backward-chaining when the target is 24 and the numbers are 12 and 2 is: "Let's work backwards from the target. 24/2 = 12. So, 12*2=24." and if the target is 22 and the numbers are 25 and 3 is: "Since the target is 22, and 22 + 3 = 25, ...".
Count the number of distinct backward-chaining instances and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any backward-chaining behavior, please provide a count of 0 as <count>0</count>."""
        ]
        return prompts


async def main():
    parser = argparse.ArgumentParser(description='Process countdown problems using OpenAI Batch API')
    parser.add_argument('--input', '-i', 
                        required=True,
                        help='Input JSON file or directory containing the countdown problems/completion files')
    parser.add_argument('--output-dir', '-o',
                        required=True,
                        help='Output directory for results')
    parser.add_argument('--api-key', '-k',
                        required=False,
                        help='OpenAI API key. If not provided, will look for OPENAI_API_KEY environment variable')
    parser.add_argument('--num-samples', '-n',
                        type=int,
                        default=200,
                        help='Number of samples to process')
    parser.add_argument('--mode', '-m',
                        choices=['batch', 'sync'],
                        default='batch',
                        help='API mode: batch (default) or sync for synchronous processing')
    parser.add_argument('--task-type', '-t',
                        choices=['countdown', 'math'],
                        default='countdown',
                        help='The type of task to process')
    args = parser.parse_args()
    
    analyzer = CountdownAnalyzer(args.api_key or os.getenv('OPENAI_API_KEY')) if args.task_type == 'countdown' else MathAnalyzer(args.api_key or os.getenv('OPENAI_API_KEY'))
    
    try:
        if args.mode == 'batch':
            print("Using batch API mode...")
            # args.input can be a file or a directory.
            batch_input_file = analyzer.convert_to_batch_format(args.input, num_samples=args.num_samples)
            file_id = analyzer.upload_file(batch_input_file)
            batch_id = analyzer.submit_batch(file_id)
            final_batch = await analyzer.poll_until_complete(batch_id)
            if final_batch.status == "completed" and final_batch.output_file_id:
                results_file = os.path.join(args.output_dir, "batch_results.jsonl")
                analyzer.download_results(final_batch.output_file_id, results_file)
                analyzer.process_results(batch_input_file, results_file, args.output_dir)
                if os.path.exists(batch_input_file):
                    os.remove(batch_input_file)
        else:
            print("Using synchronous API mode...")
            problem_metrics = analyzer.process_synchronously(args.input, num_samples=args.num_samples)
            # (The synchronous branch remains similar; you may add directory handling if desired.)
            num_problems = len(problem_metrics)
            metrics = {
                'verification_count': sum(v.get('verification_count', 0) for v in problem_metrics.values()),
                'backtracking_count': sum(v.get('backtracking_count', 0) for v in problem_metrics.values()),
                'subgoal_count': sum(v.get('subgoal_count', 0) for v in problem_metrics.values()),
                'backward_count': sum(v.get('backward_count', 0) for v in problem_metrics.values())
            }
            avg_metrics = {
                'avg_verifications': metrics['verification_count'] / num_problems,
                'avg_backtracking': metrics['backtracking_count'] / num_problems,
                'avg_subgoals': metrics['subgoal_count'] / num_problems,
                'avg_backwards': metrics['backward_count'] / num_problems
            }
            os.makedirs(args.output_dir, exist_ok=True)
            # For synchronous mode, we use a fixed step value of 1.
            detailed_path = os.path.join(args.output_dir, "evaluation_step1.json")
            with open(detailed_path, 'w') as f:
                json.dump({
                    'results': list(problem_metrics.values()),
                    'metrics': {**metrics, **avg_metrics}
                }, f, indent=2)
            aggregate_path = os.path.join(args.output_dir, "all_results.json")
            final_results = {
                'results_by_step': {
                    '1': {
                        'avg_verifications': avg_metrics['avg_verifications'],
                        'avg_backtracking': avg_metrics['avg_backtracking'],
                        'avg_subgoals': avg_metrics['avg_subgoals'],
                        'avg_backwards': avg_metrics['avg_backwards'],
                        'total_verifications': metrics['verification_count'],
                        'total_backtracking': metrics['backtracking_count'],
                        'total_subgoals': metrics['subgoal_count'],
                        'total_backwards': metrics['backward_count']
                    }
                },
                'steps_processed': ['1']
            }
            if os.path.exists(aggregate_path):
                with open(aggregate_path, 'r') as f:
                    existing_results = json.load(f)
                existing_results['results_by_step']['1'] = final_results['results_by_step']['1']
                existing_results['steps_processed'] = sorted(set(existing_results['steps_processed'] + ['1']))
                final_results = existing_results
            with open(aggregate_path, 'w') as f:
                json.dump(final_results, f, indent=2)
            print(f"\nResults saved to {detailed_path} and {aggregate_path}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
