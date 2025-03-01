export N_GPUS=4
export BASE_MODEL=./countdown_qwen2.5-3b_backtracking_subgoal_sft/global_step_60
export DATA_DIR=./countdown
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown_qwen2.5-3b_backtracking_subgoal_ppo
export VLLM_ATTENTION_BACKEND=XFORMERS

sh ./scripts/train_tiny_zero_n4_cd.sh
