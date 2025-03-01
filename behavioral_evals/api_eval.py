import json
import random
from google import genai
from google.genai import types
from argparse import ArgumentParser
import os
import re
from typing import List, Dict, Any
import asyncio
from datetime import datetime
from tqdm import tqdm

class RateLimitedEvaluator:
    def __init__(self, api_key: str, rate_limit: int = 50, target_utilization: float = 0.9):
        self.client = genai.Client(api_key=api_key)
        self.rate_limit = rate_limit  # requests per minute
        self.target_utilization = target_utilization
        self.target_requests_per_minute = rate_limit * target_utilization
        self.request_times = []
        self.last_request_time = None
        self.target_interval = 60 / self.target_requests_per_minute
        # A lock to protect rate-limit shared state when running concurrently.
        self.rate_limit_lock = asyncio.Lock()

    async def _maintain_request_rate(self):
        """Maintain a steady request rate at target utilization with a simplified hand‐brake."""
        async with self.rate_limit_lock:
            now = datetime.now()
            
            if self.last_request_time is not None:
                # Calculate time since last request
                elapsed = (now - self.last_request_time).total_seconds()
                # If we're going faster than our target rate, wait
                if elapsed < self.target_interval:
                    await asyncio.sleep(self.target_interval - elapsed)
            
            # Update last request time after any necessary wait
            self.last_request_time = datetime.now()
            
            # Remove entries older than 1 minute
            self.request_times = [t for t in self.request_times if (now - t).total_seconds() < 60]
            self.request_times.append(now)
            
            # Emergency brake: if we've reached the rate limit, wait until the oldest request is over a minute old
            if len(self.request_times) >= self.rate_limit:
                oldest = self.request_times[0]
                elapsed_since_oldest = (datetime.now() - oldest).total_seconds()
                wait_time = max(0, 60 - elapsed_since_oldest)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                # After waiting, clear the request history
                self.request_times = []

    async def _make_rate_limited_request(self, prompt: str) -> str:
        """Make a rate-limited request to the Gemini API with hand‐break applied for errors."""
        max_retries = 3
        attempts = 0
        while True:
            await self._maintain_request_rate()
            try:
                response = self.client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.,
                        top_p=0.95,
                        top_k=64,
                        max_output_tokens=256,
                        response_mime_type="text/plain"
                    )
                )
                if response.text is None:
                    raise Exception("None response")
                return response.text
            except Exception as e:
                error_message = str(e)
                if (
                    "429" in error_message or 
                    "RESOURCE_EXHAUSTED" in error_message or
                    "None response" in error_message or 
                    "500" in error_message or 
                    "INTERNAL" in error_message
                ):
                    attempts += 1
                    print(
                        f"Received RESOURCE_EXHAUSTED or INTERNAL error. "
                        f"Attempt {attempts} of {max_retries}. "
                        f"Applying hand‐break and retrying lost example."
                    )
                    await asyncio.sleep(60)
                    # Reset request history after a hand‐break
                    self.request_times = []
                    if attempts >= max_retries:
                        raise e
                else:
                    raise e

    async def _verify_with_gemini(
        self, query: str, completion: str, target: int = None, numbers: List[int] = None
    ) -> Dict[str, Any]:
        """Run four checks on the model's chain-of-reasoning:
           1) Answer verification steps
           2) Backtracking behavior
           3) Subgoal setting
           4) Backward-chaining
        """
        # Extract numbers and target from query if not provided
        if target is None:
            target_match = re.search(r"results in (\d+)", query)
            if target_match:
                target = int(target_match.group(1))

        if numbers is None:
            numbers_match = re.search(r"numbers ([\d, ]+)", query)
            if numbers_match:
                numbers = [int(x.strip()) for x in numbers_match.group(1).split(",")]

        # Default fallback if not parseable:
        if not numbers:
            numbers = []
        if not target:
            target = 0

        # Regex pattern for extracting the count from Gemini's response
        count_match = r"<count>\s*(\d+)\s*</count>"

        # 1. Answer-verification steps
        verification_prompt = f"""Here is a chain-of-reasoning that a Language Model generated while trying to play the game of CountDown with the numbers {numbers}. The goal is to reach the target number {target}. The chain-of-reasoning the model used is: {completion}. 

Evaluate whether the chain-of-reasoning contains any answer-verification steps. An example of an answer-verification step is: 'This sequence results in 1, which is not equal to 22' or 'Since 25 is not equal to 22'. We want to mark instances where the chain-of-reasoning explicitly checks the current result against the target number. 

If you find any answer-verification steps, please count them and provide the count as between the tags <count> </count>. If the chain-of-reasoning does not contain any answer-verification steps, please provide a count of 0 as <count>0</count>."""
        verification_response = await self._make_rate_limited_request(verification_prompt)
        try:
            verification_count = int(re.search(count_match, verification_response).group(1))
        except:
            verification_count = 0

        # 2. Backtracking behavior
        backtracking_prompt = f"""Here is a chain-of-reasoning that a Language Model generated while trying to play the game of CountDown with the numbers {numbers}. The goal is to reach the target number {target}. The chain-of-reasoning the model used is: {completion}.

Evaluate whether the chain-of-reasoning contains any backtracking behavior, where the model realizes a path won't work and explicitly goes back to try a different approach. An example of backtracking is: "Let me try again" or "we need to try a different sequence". We want to mark instances where the chain-of-reasoning is abandoned and the model backtracks to a previous computation. 

Count the number of distinct backtracking instances and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any backtracking behavior, please provide a count of 0 as <count>0</count>."""
        backtracking_response = await self._make_rate_limited_request(backtracking_prompt)
        try:
            backtracking_count = int(re.search(count_match, backtracking_response).group(1))
        except:
            backtracking_count = 0

        # 3. Subgoal setting
        subgoal_prompt = f"""Here is a chain-of-reasoning that a Language Model generated while trying to play the game of CountDown with the numbers {numbers}. The goal is to reach the target number {target}. The chain-of-reasoning the model used is: {completion}.

Evaluate whether the chain-of-reasoning contains any explicit subgoal setting, where the model breaks down the problem into smaller, intermediate goals. An example of subgoal setting is: "First, I'll try to get close to {target//2}, then...".

Count the number of distinct subgoals set and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any subgoal setting, please provide a count of 0 as <count>0</count>."""
        subgoal_response = await self._make_rate_limited_request(subgoal_prompt)
        try:
            subgoal_count = int(re.search(count_match, subgoal_response).group(1))
        except:
            subgoal_count = 0

        # 4. Backward-chaining
        backward_chaining_prompt = f"""Here is a chain-of-reasoning that a Language Model generated while trying to play the game of CountDown with the numbers {numbers}. The goal is to reach the target number {target}. The chain-of-reasoning the model used is: {completion}.

Evaluate whether the chain-of-reasoning contains any backward-chaining behavior, where the model starts from the target number and works backwards to the initial numbers. An example of backward-chaining when the target is 24 and the numbers are 12 and 2 is: "Let's work backwards from the target. 24/2 = 12. So, 12*2=24." and if the target is 22 and the numbers are 25 and 3 is: "Since the target is 22, and 22 + 3 = 25, ...".

Count the number of distinct backward-chaining instances and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any backward-chaining behavior, please provide a count of 0 as <count>0</count>."""
        backward_response = await self._make_rate_limited_request(backward_chaining_prompt)
        try:
            backward_count = int(re.search(count_match, backward_response).group(1))
        except:
            backward_count = 0
        
        return {
            'verification_count': verification_count,
            'backtracking_count': backtracking_count,
            'subgoal_count': subgoal_count,
            'backward_count': backward_count,
        }

    async def process_completions(self, completions: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        # Limit concurrency with a semaphore
        semaphore = asyncio.Semaphore(100)
        results = []
        
        async def process_item(item):
            async with semaphore:
                try:
                    # The script supports either 'query'/'completion' fields or 'prompt'/'generated'
                    prompt = item['query'] if 'query' in item else item['prompt']
                    completion = item['completion'] if 'completion' in item else item['generated']
                    target = item.get('ground_truth', {}).get('target', None)
                    numbers = item.get('ground_truth', {}).get('numbers', None)
                    result = await self._verify_with_gemini(prompt, completion, target, numbers)
                    return {
                        'query': prompt,
                        'completion': completion,
                        **result
                    }
                except Exception as e:
                    print(f"Error processing item: {e}")
                    return None
        
        tasks = [asyncio.create_task(process_item(item)) for item in completions]
        # Use asyncio.as_completed with tqdm to track progress
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing completions"):
            res = await future
            if res is not None:
                results.append(res)
        return results

def get_completion_files(completions_dir: str) -> List[tuple[int, str]]:
    """Get sorted list of completion files with their step numbers."""
    files = []
    for filename in os.listdir(completions_dir):
        if filename.startswith('completions_step') and filename.endswith('.jsonl'):
            match = re.search(r'completions_step(\d+)\.jsonl', filename)
            if match:
                step = int(match.group(1))
                files.append((step, os.path.join(completions_dir, filename)))
    return sorted(files)  # Sort by step number

async def main():
    parser = ArgumentParser(description="Evaluate completions using Gemini API")
    parser.add_argument(
        "--completions-dir",
        type=str,
        required=True,
        help="Directory containing completion JSONL files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store evaluation results"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=GEMINI_API_KEY,
        help="Gemini API key"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of random completions to sample from each file. If not set or 0, all completions are processed."
    )
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all completion files
    if not args.completions_dir.endswith('.jsonl'):
        completion_files = get_completion_files(args.completions_dir)
    else:
        if 'step' in args.completions_dir:
            step = int(re.search(r'step_?(\d+)', args.completions_dir).group(1))
        else:
            step = 0
        completion_files = [(step, args.completions_dir)]
    print(f"Found {len(completion_files)} completion file(s).")
    
    # Initialize evaluator
    evaluator = RateLimitedEvaluator(args.api_key)
    
    # Process each file
    all_results = {}
    for step, filepath in completion_files:
        print(f"\nProcessing step {step} from {os.path.basename(filepath)}")
        
        # Load completions
        with open(filepath, 'r') as f:
            completions = json.load(f)

        # Optionally sample completions if --num-samples is provided
        if args.num_samples and args.num_samples > 0:
            if len(completions) > args.num_samples:
                completions = random.sample(completions, args.num_samples)
                print(f"Randomly sampled {args.num_samples} completions.")
            else:
                print(f"Requested {args.num_samples} samples, but only {len(completions)} items found. Using all items.")
        else:
            print(f"Using all {len(completions)} completions from file.")
        
        # Process completions concurrently
        print(f"Processing {len(completions)} completions...")
        results = await evaluator.process_completions(completions)
        
        # Calculate metrics for this step
        total_verifications = sum(r['verification_count'] for r in results)
        total_backtracking = sum(r['backtracking_count'] for r in results)
        total_subgoals = sum(r['subgoal_count'] for r in results)
        total_backwards = sum(r['backward_count'] for r in results)
        
        avg_verifications = total_verifications / len(results) if results else 0
        avg_backtracking = total_backtracking / len(results) if results else 0
        avg_subgoals = total_subgoals / len(results) if results else 0
        avg_backwards = total_backwards / len(results) if results else 0
        
        # Store results for this step
        step_results = {
            'results': results,
            'metrics': {
                'total_verifications': total_verifications,
                'total_backtracking': total_backtracking,
                'total_subgoals': total_subgoals,
                'total_backwards': total_backwards,
                'avg_verifications': avg_verifications,
                'avg_backtracking': avg_backtracking,
                'avg_subgoals': avg_subgoals,
                'avg_backwards': avg_backwards
            }
        }
        
        # Save individual step results
        output_path = os.path.join(
            args.output_dir,
            f"evaluation_step{step}.json"
        )
        
        print(f"Storing results at: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(step_results, f, indent=2)
            
        # Store in aggregate results
        all_results[step] = {
            'avg_verifications': avg_verifications,
            'avg_backtracking': avg_backtracking,
            'avg_subgoals': avg_subgoals,
            'avg_backwards': avg_backwards,
            'total_verifications': total_verifications,
            'total_backtracking': total_backtracking,
            'total_subgoals': total_subgoals,
            'total_backwards': total_backwards,
        }
        
        print(f"\nStep {step} complete:")
        print(f"  Average verifications: {avg_verifications:.4f}")
        print(f"  Average backtracking:  {avg_backtracking:.4f}")
        print(f"  Average subgoals:      {avg_subgoals:.4f}")
        print(f"  Average backwards:     {avg_backwards:.4f}")
        print(f"  Total verifications:   {total_verifications}")
        print(f"  Total backtracking:    {total_backtracking}")
        print(f"  Total subgoals:        {total_subgoals}")
        print(f"  Total backwards:       {total_backwards}")
    
    # Save aggregate results
    # First read existing results if the file exists
    # First read existing results if the file exists
    aggregate_path = os.path.join(args.output_dir, "all_results.json")
    existing_results = {"results_by_step": {}, "steps_processed": []}
    
    if os.path.exists(aggregate_path):
        with open(aggregate_path, 'r') as f:
            existing_results = json.load(f)
    
    # Convert new results keys to strings
    all_results = {str(step): results for step, results in all_results.items()}
    
    # Merge new results with existing ones
    # Only add results for steps that aren't already present
    for step, results in all_results.items():
        if step not in existing_results['results_by_step']:
            existing_results['results_by_step'][step] = results
    
    # Update the steps processed list - convert to integers for sorting, then back to strings
    existing_results['steps_processed'] = [
        str(x) for x in sorted(int(x) for x in existing_results['results_by_step'].keys())
    ]
    
    # Write back the merged results
    with open(aggregate_path, 'w') as f:
        json.dump(existing_results, f, indent=2)
    
    print("\nAll processing complete!")
    print(f"Full results saved to: {aggregate_path}")

if __name__ == "__main__":
    asyncio.run(main())
