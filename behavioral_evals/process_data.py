#!/usr/bin/env python3
import os
import json
import csv
import sys
import argparse

def load_all_results(root_dir, single=False):
    """
    Walks the immediate subdirectories of root_dir.
    For each subdirectory, if an 'all_results.json' file exists,
    it is loaded via json.load.
    
    Returns a dictionary mapping condition names (the subdirectory name)
    to a dictionary with keys:
      - "all_results": the parsed results from all_results.json
      - "dir": the full path to that subdirectory
    """
    results = {}
    if not single:
        for entry in os.listdir(root_dir):
            subdir = os.path.join(root_dir, entry)
            if os.path.isdir(subdir):
                json_path = os.path.join(subdir, "all_results.json")
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                        results[entry] = {"all_results": data, "dir": subdir}
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in {json_path}: {e}", file=sys.stderr)
                    except Exception as e:
                        print(f"Error reading {json_path}: {e}", file=sys.stderr)
                else:
                    print(f"Warning: {json_path} does not exist.", file=sys.stderr)
    else:
        json_path = os.path.join(root_dir, "all_results.json")
        condition_str = root_dir.split('/')[-1]
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                results[condition_str] = {"all_results": data, "dir": root_dir}
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {json_path}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Error reading {json_path}: {e}", file=sys.stderr)
        else:
            print(f"Warning: {json_path} does not exist.", file=sys.stderr)
    return results

def read_completions(completions_path):
    """
    Reads the completions file from completions_path.
    The expected format is a JSON array (or a JSON lines file that can be read with json.load).
    Returns the list of completion entries, or an empty list if reading fails.
    """
    if not os.path.exists(completions_path):
        print(f"Warning: Completions file {completions_path} does not exist.", file=sys.stderr)
        return []
    try:
        with open(completions_path, 'r') as f:
            # Try to load the file as JSON.
            completions = json.load(f)
        return completions
    except Exception as e:
        print(f"Error reading completions from {completions_path}: {e}", file=sys.stderr)
        return []

def compute_completions_metrics(completions):
    """
    Given a list of completion entries, computes:
      - average accuracy (from the "score" field)
      - average response length (number of words in the "generated" field)
    If a particular field is missing in an entry, that entry is skipped for that metric.
    Returns a tuple: (avg_accuracy, avg_response_length)
    """
    accs = []
    resp_lengths = []
    for comp in completions:
        if "score" in comp and isinstance(comp["score"], (int, float)):
            accs.append(comp["score"])
        if "generated" in comp and isinstance(comp["generated"], str):
            resp_lengths.append(len(comp["generated"]))
    avg_acc = sum(accs)/len(accs) if accs else None
    avg_resp_length = sum(resp_lengths)/len(resp_lengths) if resp_lengths else None
    return avg_acc, avg_resp_length

def process_data(results):
    """
    Given the collated results dictionary (mapping condition -> {"all_results": ..., "dir": ...}),
    iterate over each condition and each step (only include steps that are strings) in the condition's
    "steps_processed" list.
    
    For each such step:
      - Retrieve the metric values from the "results_by_step" dictionary.
      - Look for the corresponding completions file: completions_step{step}.jsonl in that condition's directory.
      - Compute the average accuracy and average response length from the completions file.
      - Return a list of flattened row dictionaries.
    """
    rows = []
    for condition, info in results.items():
        cond_data = info.get("all_results", {})
        cond_dir = info.get("dir", "")
        steps = cond_data.get("steps_processed", [])
        results_by_step = cond_data.get("results_by_step", {})
        for step in steps:
            if not isinstance(step, str):
                continue  # Only process steps that are strings.
            if step not in results_by_step:
                print(f"Warning: step '{step}' not found in results_by_step for condition '{condition}'.", file=sys.stderr)
                continue
            # Get the metrics from the all_results.json
            metrics = results_by_step[step]
            # Build the base row.
            row = {"condition": condition, "step": step}
            row.update(metrics)
            # Now, locate and process the completions file.
            if int(step) == 1: 
                step = '0'
            completions_filename = f"completions_step{step}.jsonl"
            completions_path = os.path.join(cond_dir, completions_filename)
            completions = read_completions(completions_path)
            avg_acc, avg_resp_length = compute_completions_metrics(completions)
            row["accuracy"] = avg_acc
            row["response_length"] = avg_resp_length
            rows.append(row)
    breakpoint()
    return rows

def main():
    parser = argparse.ArgumentParser(
        description="Collate all_results.json files from a folder, extract per-step metrics "
                    "and also compute accuracy and response length from completions files. "
                    "Flatten the data into a CSV dataset."
    )
    parser.add_argument("input",
                        help="Input folder containing condition subdirectories (e.g., 'outputs')")
    parser.add_argument("--output", "-o", required=True,
                        help="Output CSV file for the flattened dataset")
    parser.add_argument("--single", "-s", action="store_true",
                        help="Use this flag to process a single condition")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Error: {args.input} is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    # Collate the results from each condition.
    results = load_all_results(args.input, args.single)
    if not results:
        print("No results were loaded. Check that your subdirectories contain all_results.json files.",
              file=sys.stderr)
        sys.exit(1)

    # Process the collated data (including completions metrics) to flatten it.
    rows = process_data(results)
    if not rows:
        print("No rows were produced. Check that your input data has steps_processed as strings.",
              file=sys.stderr)
        sys.exit(1)

    # Determine the fieldnames from the first row.
    fieldnames = list(rows[0].keys())
    # Write the flattened data to CSV.
    try:
        with open(args.output, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if csvfile.tell() == 0:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"Appended {len(rows)} rows to {args.output}")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()