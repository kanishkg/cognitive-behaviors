import json
import re
import argparse


def remove_think_content(text):
    # Pattern to match content between <think> and </think> tags
    pattern = r'<think>.*?</think>'
    # Replace with empty think tags
    return re.sub(pattern, '<think> </think>', text, flags=re.DOTALL)

def process_jsonl(input_file, output_file):
    # Read entire JSON array
    with open(input_file, 'r') as infile:
        data = json.load(infile)
    
    # Process all objects
    for item in data:
        if 'completion' in item:
            item['completion'] = remove_think_content(item['completion'])
    
    # Write processed data
    with open(output_file, 'w') as outfile:
        json.dump(data, outfile, indent=2)

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