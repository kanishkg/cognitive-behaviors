import os
import re

import datasets

def extract_variables(markdown_text):
    """
    Extracts variable values from a markdown string.
    The markdown is expected to have sections where a header (starting with '##')
    is immediately followed (after optional blank lines) by a single line
    containing the variable value.
    
    Args:
        markdown_text (str): The markdown content as a string.
        
    Returns:
        dict: A dictionary where keys are header texts and values are the extracted variable lines.
    """
    # This regex works as follows:
    # - It finds a header line that starts with '##', capturing any text that follows.
    # - It then matches one or more newline characters (allowing for blank lines),
    #   and captures the first non-empty line that follows as the variable value.
    pattern = r"##\s*(.*?)\s*\n+(?!##)([^\n]+)"
    
    matches = re.findall(pattern, markdown_text)
    
    variables = {}
    for header, var in matches:
        # Strip any extra whitespace from the variable text
        variables[header] = var.strip()
    return variables

def map_fn_backtrack(examples):
    ret_dict = {}
    # Process each sample in the batch
    for i in range(len(examples['text'])):
        curr_backtrack = examples['backtracking_raw'][i]
        count = re.findall(r'<count>(\d+)</count>', curr_backtrack)
        if count:
            count = int(count[0])
        else:
            count = None
        ret_dict.setdefault('backtrack_count', []).append(count)
        for k in examples.keys():
            ret_dict.setdefault(k, []).append(examples[k][i])
            
    return ret_dict

def map_fn_backchain(examples):
    ret_dict = {}
    for i in range(len(examples['text'])):
        curr_backchain = examples['backward_chaining_raw'][i]
        count = re.findall(r'<count>(\d+)</count>', curr_backchain)
        if count:
            count = int(count[0])
        else:
            count = None
        ret_dict.setdefault('backchain_count', []).append(count)
        
        for k in examples.keys():
            ret_dict.setdefault(k, []).append(examples[k][i])
            
    return ret_dict

def map_fn_verification(examples):
    ret_dict = {}
    for i in range(len(examples['text'])):            
        curr_verification = examples['verification_raw'][i]
        count = re.findall(r'<count>(\d+)</count>', curr_verification)
        if count:
            count = int(count[0])
        else:
            count = None
        ret_dict.setdefault('verification_count', []).append(count)
        for k in examples.keys():
            ret_dict.setdefault(k, []).append(examples[k][i])
            
    return ret_dict

def map_fn_subgoal(examples):
    ret_dict = {}
    for i in range(len(examples['text'])):
        curr_subgoal = examples['subgoal_setting_raw'][i]
        # extract count using xml tags, regex <count>(\d+)</count>
        count = re.findall(r'<count>(\d+)</count>', curr_subgoal)
        if count:
            count = int(count[0])
        else:
            count = None
        ret_dict.setdefault('subgoal_count', []).append(count)
        for k in examples.keys():
            ret_dict.setdefault(k, []).append(examples[k][i])
            
    return ret_dict

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='', help='Dataset name')
args = parser.parse_args()

dataset_name = args.dataset_name
assert dataset_name is not None, "Dataset name is not set"
ds = datasets.load_dataset(dataset_name, split='train')

# Apply the mapping functions
ds = ds.map(map_fn_backtrack, batched=True, remove_columns=ds.column_names, num_proc=64)
ds = ds.map(map_fn_backchain, batched=True, remove_columns=ds.column_names, num_proc=64)
ds = ds.map(map_fn_verification, batched=True, remove_columns=ds.column_names, num_proc=64)
ds = ds.map(map_fn_subgoal, batched=True, remove_columns=ds.column_names, num_proc=64)

ds.push_to_hub(dataset_name+'_processed')



