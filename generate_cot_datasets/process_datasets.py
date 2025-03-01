#!/usr/bin/env python3
import json
import argparse
import tempfile
import shutil

def process_jsonl_file(input_file):
    """
    Process a JSONL file to change 'prompt' key to 'query' in each JSON object
    
    Args:
        input_file (str): Path to the input JSONL file
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            # Process the input file line by line
            with open(input_file, 'r') as fin:
                for line_num, line in enumerate(fin, 1):
                    try:
                        # Parse JSON object
                        data = json.loads(line.strip())
                        
                        # Replace 'prompt' with 'query' if it exists
                        if 'prompt' in data:
                            data['query'] = data.pop('prompt')
                        
                        # Write the modified JSON object to temp file
                        json.dump(data, temp_file)
                        temp_file.write('\n')
                        
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON on line {line_num}: {e}")
                        continue

        # Replace the original file with the temporary file
        shutil.move(temp_file.name, input_file)
        print(f"Processing complete. File '{input_file}' has been updated.")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except PermissionError:
        print(f"Error: Permission denied when accessing '{input_file}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert "prompt" key to "query" in JSONL file')
    parser.add_argument('filename', help='Path to the input JSONL file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the file
    process_jsonl_file(args.filename)

if __name__ == "__main__":
    main()