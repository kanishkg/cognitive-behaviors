#!/usr/bin/env python3
import json
import os
import glob
from typing import Dict, Any

def aggregate_evaluation_results(output_dir: str) -> None:
    """
    Aggregate all evaluation step results into a single all_results.json file.
    
    Args:
        output_dir: Directory containing the evaluation step files
    """
    # Initialize the aggregate results structure
    aggregate_results = {
        "results_by_step": {},
        "steps_processed": []
    }
    
    # Find all evaluation step files
    pattern = os.path.join(output_dir, "evaluation_step*.json")
    step_files = glob.glob(pattern)

    step_files = [(step_file.split("step")[-1].split(".")[0], step_file) for step_file in step_files]
    step_files = sorted(step_files, key=lambda x: int(x[0]))
    
    # Process each step file
    for step, step_file in step_files:
        try:
            # Read the step results
            with open(step_file, 'r') as f:
                step_data = json.load(f)
            
            # Extract the metrics
            metrics = step_data['metrics']
            
            # Create the step entry
            step_entry = {
                "avg_verifications": metrics['avg_verifications'],
                "avg_backtracking": metrics['avg_backtracking'],
                "avg_subgoals": metrics['avg_subgoals'],
                "avg_backwards": metrics['avg_backwards'],
                "total_verifications": metrics['verification_count'],
                "total_backtracking": metrics['backtracking_count'],
                "total_subgoals": metrics['subgoal_count'],
                "total_backwards": metrics['backward_count']
            }
            
            # Add to aggregate results
            aggregate_results['results_by_step'][step] = step_entry
            
        except Exception as e:
            print(f"Error processing {step_file}: {str(e)}")
            continue
    
    # Update steps processed list
    numeric_steps = [s for s in aggregate_results['results_by_step'].keys() 
                    if s.isdigit()]
    aggregate_results['steps_processed'] = sorted(numeric_steps, key=int)
    
    # Save the aggregate results
    output_path = os.path.join(output_dir, "all_results.json")
    with open(output_path, 'w') as f:
        json.dump(aggregate_results, f, indent=2)
    
    print(f"Aggregated results saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate evaluation results from step files")
    parser.add_argument("output_dir", help="Directory containing the evaluation step files")
    
    args = parser.parse_args()
    
    aggregate_evaluation_results(args.output_dir)