import anthropic
import argparse
import uuid
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from enum import Enum
import random
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from tasks.countdown import CountDown

class DatasetType(Enum):
    NEGATIVE_CONTROL = "negative_control"
    ONLY_BACKTRACKING = "only_backtracking"
    BACKTRACKING_VERIFICATION = "backtracking_verification"
    BACKTRACKING_BACKWARD = "backtracking_backward"
    BACKTRACKING_SUBGOAL = "backtracking_subgoal"
    ALL_STRATEGIES = "all_strategies"

@dataclass
class SystemPrompt:
    prompt_text: str
    cache_control: Dict

class DatasetGenerator:
    def __init__(self, api_key: str, dataset_type: DatasetType, 
                 target_samples: int, output_file: str, 
                 seed: int = 42, max_target: int = 25, min_target: int = 10):
        self.api_key = api_key
        self.client = anthropic.Anthropic(api_key=api_key)
        self.dataset_type = dataset_type
        self.target_samples = target_samples
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize CountDown task generator
        random.seed(seed)
        self.countdown = CountDown(
            start_probs=[0., 0.5, 0.5],
            # start_probs=[0.0, 0.0, 1.],
            max_target=max_target,
            min_target=min_target
        )
        
        # Initialize system prompts
        self.system_prompts = {
            DatasetType.NEGATIVE_CONTROL: SystemPrompt(
                prompt_text="""I want to produce reasoning trajectories for the game of countdown. The goal here is to reach a target number by combining integers using basic arithmetic operations.
The answer is a series of arithmetic operations (+, -, *, /) that results in the target number.
Write your answers in <think> </think> tags. Your thinking should be of the form <think> Step X: number1 (+,-,*,/) number2 = result. </think>
Write the final answer in <answer> </answer> tags.
For the final answer, make sure that each step in the final answer is written as <answer> (number1 [+-*/] number2) [+-*/] number3 </answer>.
Answer should be a valid mathematical expression ONLY containing starting integers and NOT the target number. 
Otherwise, the grader will not be able to parse your answer.
Here are some rules:
- DO NOT backtrack, directly give me the correct answer.
- DO NOT work backwards from the goal.
- DO NOT explicitly verify your answer.
- DIRECTLY give me the solution.
- DO NOT produce any other outputs.""",
                cache_control={"type": "ephemeral"}
            ),
            
            DatasetType.ONLY_BACKTRACKING: SystemPrompt(
                prompt_text="""I want to produce reasoning trajectories for the game of countdown. The goal here is to reach a target number by combining integers using basic arithmetic operations.
Write your thoughts in <think> </think> tags.
The answer is a series of arithmetic operations (+, -, *, /) that results in the target number.
Write the final answer in <answer> </answer> tags.
For the final answer, make sure that each step in the final answer is written as <answer> (number1 [+-*/] number2) [+-*/] number3 </answer>.
Answer should be a valid mathematical expression ONLY containing starting integers and NOT the target number.
Otherwise, the grader will not be able to parse your answer.
Backtrack to the start or an intermediate step if you haven't reached the answer.
- DO NOT set subgoals.
- DO NOT work backwards from the goal.
- DO NOT explicitly verify your answer.
For example, you can say things like:
1. When the target is 24 and you have [12, 2]: "12+2 = 14. Trying something else. 12*2=24." You are not allowed say things like "14 is not 24" or "14 is close to 24" or "14 is too high". Omit evaluations in text.
2. When the target is 10 and you have [12, 2]: "12+2 = 14. Let's try a different sequence of operations.\"""",
                cache_control={"type": "ephemeral"}
            ),
            
            DatasetType.BACKTRACKING_VERIFICATION: SystemPrompt(
                prompt_text="""I want to produce reasoning trajectories for the game of countdown. The goal here is to reach a target number by combining integers using basic arithmetic operations.
Write your thoughts in <think> </think> tags.
The answer is a series of arithmetic operations (+, -, *, /) that results in the target number.
Write the final answer in <answer> </answer> tags.
For the final answer, make sure that each step in the final answer is written as <answer> (number1 [+-*/] number2) [+-*/] number3 </answer>.
Answer should be a valid mathematical expression ONLY containing starting integers and NOT the target number.
Otherwise, the grader will not be able to parse your answer.
Verify that you have reached the answer and backtrack to the start or an intermediate step. DO NOT set subgoals or work backwards from the goal.
For example, you can say things like:
1. When the target is 24 and you have [12, 2]: "12+2 = 14. 14 is not 24, so let's try something else. 12*2=24 and 24 was the goal, so the goal has been reached."
2. When the target is 10 and you have [12, 2]: "12+2 = 14. 14 is not 10, let's try a different sequence of operations.\"""",
                cache_control={"type": "ephemeral"}
            ),
            
            DatasetType.BACKTRACKING_BACKWARD: SystemPrompt(
                prompt_text="""I want to produce reasoning trajectories for the game of countdown. The goal here is to reach a target number by combining integers using basic arithmetic operations.
Write your thoughts in <think> </think> tags.
The answer is a series of arithmetic operations (+, -, *, /) that results in the target number.
Write the final answer in <answer> </answer> tags.
For the final answer, make sure that each step in the final answer is written as <answer> (number1 [+-*/] number2) [+-*/] number3 </answer>.
Answer should be a valid mathematical expression ONLY containing starting integers and NOT the target number.
Otherwise, the grader will not be able to parse your answer.
Backtrack to the start or an intermediate step if you haven't reached the answer. Work backwards from the goal if it makes things easier.
- DO NOT set subgoals.
- DO NOT explicitly verify your answer.
For example, you can say things like:
1. For backward chaining, when the target is 24 and you have [12, 2]: "Let's work backwards from the target. 24/2 = 12. So, 12*2=24."
2. You are not allowed say things like "14 is not 24" or "14 is close to 24" or "14 is too high". Omit evaluations in text.
3. For backtracking, when the target is 10 and you have [12, 2]: "12+2 = 14. Let's try a different sequence of operations."
4. Backward chaining might be useful to get to input numbers that you already have.
5. Do not use text evaluations like "it is too small" or "it is too high" or "this is not the target"
6. Ensure that you try working backwards from the target. If working backwards does not work, work forward, interleave the two.""",
                cache_control={"type": "ephemeral"}
            ),
            
            DatasetType.BACKTRACKING_SUBGOAL: SystemPrompt(
                prompt_text="""I want to produce reasoning trajectories for the game of countdown. The goal here is to reach a target number by combining integers using basic arithmetic operations.
Write your thoughts in <think> </think> tags.
The answer is a series of arithmetic operations (+, -, *, /) that results in the target number.
Write the final answer in <answer> </answer> tags.
For the final answer, make sure that each step in the final answer is written as <answer> (number1 [+-*/] number2) [+-*/] number3 </answer>.
Answer should be a valid mathematical expression ONLY containing starting integers and NOT the target number.
Otherwise, the grader will not be able to parse your answer.
Decompose the answer into sub-goals and try to reach them to then reach the target, if you are unable to reach the goal or a subgoal backtrack to a previous state.
- DO NOT work backwards from the goal.
- DO NOT explicitly verify your answer.
For example, you can say things like:
HINT: Set subgoals that are useful like factors of the target or multiples of the target. Or numbers close to the target.
1. When the target is 24 and you have [10, 2, 2]: "Let's first try to reach 12 since it is a factor of 24; 10 * 2 = 20, let's try a different sequence. 10 + 2 = 12. Now, 12 * 2 = 24." You are not allowed say things like "14 is not 24" or "14 is close to 24" or "14 is too high". Omit evaluations in text.
2. When the target is 10 and you have [9, 3, 2]: "Let's try to reach 20 since it is a multiple of 10…" If you can't reach it, then try something else.
3. Explicitly state the sub-goal you are trying to reach.
4. The sub-goal should not be the target number. 
5. Do not use text evaluations like "it is too small" or "it is too high" or "this is not the target"
6. Examples of good subgoals could be factors of the number, or numbers close to the target number.""",
                cache_control={"type": "ephemeral"}
            ),
            
            DatasetType.ALL_STRATEGIES: SystemPrompt(
                prompt_text="""I want to produce reasoning trajectories for the game of countdown. The goal here is to reach a target number by combining integers using basic arithmetic operations.
Write your thoughts in <think> </think> tags.
The answer is a series of arithmetic operations (+, -, *, /) that results in the target number.
Write the final answer in <answer> </answer> tags.
For the final answer, make sure that each step in the final answer is written as <answer> (number1 [+-*/] number2) [+-*/] number3 </answer>.
Answer should be a valid mathematical expression ONLY containing starting integers and NOT the target number.
Otherwise, the grader will not be able to parse your answer.
- Verify that you have reached the answer and backtrack to the start or an intermediate step.
- Work backwards from the goal if it makes things easier.
- Decompose the answer into sub-goals and try to reach them to then reach the target, if you are unable to reach the goal or a subgoal backtrack to a previous state.
HINT: Set subgoals that are useful like factors of the target or multiples of the target. Or numbers close to the target.
For example, you can say things like:
1. When the target is 24 and you have [12, 2]: "12+2 = 14. 14 is not 24, so let's try something else. 12*2=24 and 24 was the goal, so the goal has been reached."
2. When the target is 10 and you have [12, 2]: "12+2 = 14. 14 is not 10, let's try a different sequence of operations."
3. When the target is 10 and you have [9, 3, 2]: "Let's try to reach 20 since it is a multiple of 10…" If you can't reach it, then try something else.
4. When the target is 24 and you have [10, 2, 2]: "Let's first try to reach 12 since it is a factor of 24; 10 * 2 = 20, let's try a different sequence. 10 + 2 = 12. Now, 12 * 2 = 24."
5. For backward chaining, when the target is 24 and you have (12, 2): "Let's work backwards from the target. 24/2 = 12. So, 12*2=24." This is useful when setting subgoals.""",
                cache_control={"type": "ephemeral"}
            )
        }

    def create_batch_requests(self):
        """Create batch requests with the appropriate system prompt"""
        system_prompt = self.system_prompts[self.dataset_type]
        # Request 10% more than needed to account for potential failures
        num_requests = int(self.target_samples * 1.05)
        requests = []

        for _ in range(num_requests):
            task_text = self.countdown.get_task(return_raw=True)['query'][0]
            custom_id = str(uuid.uuid4())

            request = Request(
                custom_id=custom_id,
                params=MessageCreateParamsNonStreaming(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    temperature=0.5,
                    system=[
                        {
                            "type": "text",
                            "text": system_prompt.prompt_text,
                            "cache_control": system_prompt.cache_control
                        }
                    ],
                    messages=[
                        {
                            "role": "user",
                            "content": task_text,
                        }
                    ]
                )
            )

            requests.append(request)

        return requests

    def poll_batch_status(self, batch_id: str, max_poll_time_hours: int = 24):
        """Poll batch status until completion or timeout"""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=max_poll_time_hours)
        poll_interval = 60  # Start with 1 minute interval
        
        print(f"Starting to poll batch {batch_id}")
        print(f"Will poll until {end_time} or batch completion")
        
        while datetime.now() < end_time:
            try:
                batch_status = self.client.messages.batches.retrieve(batch_id)
                
                # Print current status
                counts = batch_status.request_counts
                total = sum([counts.processing, counts.succeeded, counts.errored, 
                           counts.canceled, counts.expired])
                processed = counts.succeeded + counts.errored

                if counts.errored > 0:
                    breakpoint()
                
                print(f"Status: {processed}/{total} requests processed")
                print(f"Succeeded: {counts.succeeded}, Errored: {counts.errored}, "
                      f"Canceled: {counts.canceled}, Expired: {counts.expired}")
                
                if batch_status.processing_status == "ended":
                    print("Batch processing completed!")
                    return True
                
                time.sleep(poll_interval)
                # Gradually increase polling interval, max at 5 minutes
                poll_interval = min(poll_interval * 1.5, 300)
                
            except Exception as e:
                print(f"Error polling batch status: {str(e)}")
                time.sleep(poll_interval)
        
        print("Batch polling timed out after 24 hours")
        return False

    def generate_dataset(self):
        """Generate dataset with polling"""
        print(f"Starting dataset generation for type: {self.dataset_type.value}")
        print(f"Target successful samples: {self.target_samples}")
        
        # Create and submit batch
        requests = self.create_batch_requests()
        print(f"Submitting batch with {len(requests)} requests...")


        request_map = {
            req['custom_id']: {
                "prompt": req['params']['messages'][0]['content'],
                "system": req['params']['system'][0]['text'],
            }
            for req in requests
        }
        
        # Submit batch
        message_batch = self.client.messages.batches.create(requests=requests)
        batch_id = message_batch.id
        print(f"Batch {batch_id} submitted successfully")
        
        # Poll for completion
        if not self.poll_batch_status(batch_id):
            print("Batch did not complete within 24 hours")
            return
        
        # Process results
        successful_count = 0
        with open(self.output_file, 'w') as f:
            for result in self.client.messages.batches.results(batch_id):
                if result.result.type == "succeeded" and successful_count < self.target_samples:
                    prompts = request_map[result.custom_id]
                    output_data = {
                        'query': prompts['prompt'],
                        'completion': result.result.message.content[0].text,
                    }
                    f.write(json.dumps(output_data) + '\n')
                    successful_count += 1
                    
                    if successful_count >= self.target_samples:
                        break
        
        print(f"Dataset generation completed. Generated {successful_count} samples.")

def main():
    parser = argparse.ArgumentParser(description='Generate datasets using Anthropic Batch API')
    parser.add_argument('--api_key', required=True, help='Anthropic API key')
    parser.add_argument('--dataset_type', required=True, 
                       choices=[dt.value for dt in DatasetType],
                       help='Type of dataset to generate')
    parser.add_argument('--target_samples', type=int, default=1000,
                       help='Number of successful samples to generate')
    parser.add_argument('--output_file', type=str, default='./dataset_output/results.jsonl',
                       help='Output JSONL file for successful results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for task generation')
    parser.add_argument('--max_target', type=int, default=25,
                       help='Maximum target number for countdown tasks')
    parser.add_argument('--min_target', type=int, default=10,
                       help='Minimum target number for countdown tasks')
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(
        api_key=args.api_key,
        dataset_type=DatasetType(args.dataset_type),
        target_samples=args.target_samples,
        output_file=args.output_file,
        seed=args.seed,
        max_target=args.max_target,
        min_target=args.min_target
    )
    
    generator.generate_dataset()

if __name__ == "__main__":
    main()
