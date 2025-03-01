#!/bin/bash
process_model() {
    local model_name=$1
    local task_type=$2
    local output_name="${model_name%_ppo}"
    
    local ckpt_base="./checkpoints/${model_name}"
    local output_base="./outputs/${output_name}"
    
    echo "Processing checkpoints from: ${ckpt_base}"
    echo "Outputting to: ${output_base}"
    echo "Using task-type: ${task_type}"
    
    mkdir -p "${output_base}"
    
    for step in $(seq 0 10 250); do
        local ckpt_path="${ckpt_base}/global_step_${step}"
        if [ ! -d "${ckpt_path}" ]; then
            echo "Checkpoint not found: ${ckpt_path}, skipping..."
            continue
        fi
        echo "Processing step ${step}..."
        python behavioral_evals/generate_completions.py \
            --ckpt "${ckpt_base}/global_step_${step}" \
            --dataset ./countdown/test.parquet \
            --output-path "${output_base}/" \
            --task-type "${task_type}"
            
        # rm -rf "${ckpt_base}/global_step_${step}"
    done
    
    sbatch ./scripts/gpt_submit.sh "${output_name}" "${task_type}"
}

TASK_TYPE=countdown

echo "Processing all_strategies_ppo with task-type: ${TASK_TYPE}..."
process_model "all_strategies_ppo" "${TASK_TYPE}"
