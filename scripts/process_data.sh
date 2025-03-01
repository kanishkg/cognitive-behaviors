#!/usr/bin/env bash

python3 examples/data_preprocess/countdown.py \
  --cot_dir ./cot_datasets/raw_data/all_strategies.jsonl \
  --local_dir ./cot_datasets/processed_data/all_strategies \
  --train_size 1000 \
  --test_size 200
