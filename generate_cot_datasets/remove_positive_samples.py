import json
import re
import argparse

from verl.utils.reward_score.countdown import compute_score

def process_jsonl(input_file, output_file):
    # Read entire JSON array
    with open(input_file, 'r') as infile:
        data = json.load(infile)
    
    output = []
    # Process all objects
    for item in data:
        if 'completion' in item:
            generated_text = f"Assistant:\n{item['completion']}"
            target = re.search(r'equals (\d+)', item['query']).group(1)
            numbers_pattern = r'the numbers\s*\[([^\]]+)\]'
            match_nums = re.search(numbers_pattern, item['query'])
            if match_nums:
                str_nums = match_nums.group(1)
                nums_list = [int(s.strip()) for s in str_nums.split(',')]
            else:
                nums_list = []
            
            ground_truth = {
                'target': int(target),
                'numbers': nums_list
            }
            score = compute_score(generated_text, ground_truth)
            if not bool(score):
                output.append(item)
    
    # Write processed data
    with open(output_file, 'w') as outfile:
        json.dump(output, outfile, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Process JSONL file to remove content within think tags')
    parser.add_argument('input_file', help='Input JSONL file path')
    parser.add_argument('output_file', help='Output JSONL file path')
    
    args = parser.parse_args()
    
    try:
        process_jsonl(args.input_file, args.output_file)
        print(f"Successfully processed {args.input_file} and saved to {args.output_file}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()