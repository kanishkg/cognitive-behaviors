#!/bin/bash

#SBATCH --job-name=run_train          # Job name
#SBATCH --account=cocoflops           # Account name
#SBATCH --partition=cocoflops         # Partition name
#SBATCH --time=08:00:00               # Run time (hh:mm:ss)
#SBATCH -c 10                         # Number of CPU cores
#SBATCH --gres=gpu:2                  # Number of GPUs
#SBATCH --nodes=1                     # Number of nodes
# SBATCH --nodelist=cocoflops-hgx-1   # (Optional) Specific node; uncomment if needed
#SBATCH --mem=50G                     # Memory size
#SBATCH --output=./slurm_logs/%x_%j.log  # Unique log file per job: jobName_jobID.log

set -e  # Exit immediately if a command exits with a non-zero status

source ~/.bashrc
conda activate zero

# List of dataset conditions
conditions=(
  "all_strategies",
  "backtracking_backward",
  "backtracking_subgoal",
  "backtracking_verification",
  "only_backtracking",
  "dots",
  "no_positive_cot",
  "empty_cot"
)

# Base path for dataset files
base_data_path="cot_datasets/processed_data"

# Shared training parameters
prompt_key="query"
response_key="completion"
micro_batch_size=8
train_batch_size=64
max_length=2048
model_name="meta-llama/Llama-3.2-3B"
default_hdfs_dir="hdfs"
default_local_dir="sft"
project_name="countdown-sft"
total_epochs=5
logger="['console','wandb']"

# Iterate over each condition and launch a training job
for condition in "${conditions[@]}"; do
  train_file="${base_data_path}/${condition}/train.parquet"
  val_file="${base_data_path}/${condition}/test.parquet"

  experiment_name="countdown-sft-${model_name}-${condition}"

  save_dir="${default_local_dir}/${condition}"

  echo "Running training for condition: ${condition}"
  echo "Train file: ${train_file}"
  echo "Val file:   ${val_file}"
  echo "Experiment name: ${experiment_name}"
  echo ""

  torchrun --nproc_per_node=2 -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${train_file}" \
    data.val_files="${val_file}" \
    data.prompt_key="${prompt_key}" \
    data.response_key="${response_key}" \
    data.micro_batch_size="${micro_batch_size}" \
    data.train_batch_size="${train_batch_size}" \
    data.max_length="${max_length}" \
    model.partial_pretrain="${model_name}" \
    trainer.default_hdfs_dir="${default_hdfs_dir}" \
    trainer.default_local_dir="${save_dir}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.total_epochs="${total_epochs}" \
    trainer.logger="${logger}"

  echo "--------------------------------------------------"
done

