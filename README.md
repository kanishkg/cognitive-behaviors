<div align="center">
  <h1 style="border-bottom: none;">Cognitive Behaviors that Enable Self-Improving Reasoners,</h1>
  <h2 style="border-top: none;"><em>or,</em> Four Habits of Highly Effective STaRs</h2>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/License-Apache 2.0-green" alt="License">
  <a href="https://arxiv.org/abs/2503.01307" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/arXiv-2503.01307-b31b1b.svg" alt="arXiv">
  </a>
</p>

<p align="center">
  <em>“The limits of my language mean the limits of my world.”</em><br>
  – Wittgenstein
</p>


This repository is based on TinyZero, which is a reproduction of [DeepSeek R1 Zero](https://github.com/deepseek-ai/DeepSeek-R1) in countdown built upon [veRL](https://github.com/volcengine/verl).

All datasets and models will be released here on the [HF Collection](https://huggingface.co/collections/obiwan96/cognitive-behaviors-67db58981956dcedf0c80929).

## Installation

```
conda create -n zero python=3.9
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip install ray

# verl
pip install -e .

# flash attention 2
pip install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib

# behavioral evals
pip install asynciolimiter loguru tenacity anthropic openai
```

## Countdown task
First, you can generate the Countdown dataset from the original repo.

**Data Preparation**
```
conda activate zero
python ./examples/data_preprocess/countdown.py --local_dir {path_to_your_dataset}
```

## Generate Priming Data
To generate the priming data, we use `claude-3.5-sonnet`.

### Generate Behavior Data
We generate 5 datasets with different behaviors.
```
python generate_cot_datasets/api_gen.py --api_key {your_api_key} --dataset_type {dataset_type} --target_samples {target_samples} --output_file {output_file} --seed {seed} --max_target {max_target} --min_target {min_target}
# process the data into parquet format
sh ./scripts/process_data.sh
```
### Generate Empty COT Data
We generate 2 datasets with empty COT, one that is length matched to all strategies and one that just has an empty COT.
```
python generate_cot_datasets/generate_empty_cot.py --input_file {input_file} --output_file {output_file}
```

### Generate Priming Data with only Incorrect Examples
We convert the all strategies dataset to only have incorrect examples.
```
python generate_cot_datasets/generate_no_positive_cot.py --input_file {input_file} --output_file {output_file}
```

## Training
All our experiments are run on either 8 or 4, 80GB A100s or H100s.
The number of GPUs can be changed in the script. Please see the TinyZero repo or VERL for more information on training with different compute.

## Run SFT
We run SFT on the priming data to get a new primed base model.
```
chmod +x scripts/sft.sh
./scripts/sft.sh
```

## Run PPO
We run PPO on the primed model.
```
sh ./scripts/train.sh
```

## Run Behavior Evals
We run the behavioral evals with `gpt-4o-mini` to get the results.
```
sh ./scripts/behavioral_evals.sh
```

## Label Pretraining Data
First, we label the pretraining data to get the behavior counts.
```
# Classify behaviors
python pretraining_analysis/relabel_pretrain_offline.py --user username --start 0 --end 1000000 --save_every 10000 --dataset_name {dataset_name}
# Process and get stats
python pretraining_analysis/process_pretrain_labelled.py --dataset_name {dataset_name}
python pretraining_analysis/get_stats.py --dataset_name {dataset_name}
```
### Generate Synthetic Data
We generate a new dataset with the labeled data. To train a model for a broader set of downstream tasks (like gpqa or MATH), we recommend changing the prompt and processing more files from OpenWebMath or FineMath.
```
# Label as COT
python pretraining_analysis/relabel_pretrain_qa.py --user username --start 0 --end 1000000 --save_every 10000 --dataset_name {dataset_name} --mehtod {curated or negative}
# Format as QA and save as parquet
python pretraining_analysis/generate_qa_parquets.py --dataset_name {dataset_name}
# Trim for SFT and save as parquet
python pretraining_analysis/save_parquets.py --dataset_name {dataset_name}
```
We then run SFT on the new dataset to get a new primed base model and run PPO on it.


