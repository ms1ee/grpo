#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 -m scripts.train_grpo \
 --data_files data/legal_ko.jsonl  \
 --eval_data_files data/kmmlu_law.jsonl \
 --wandb_run_name   \
 --checkpoint_dir checkpoints/ \
 --use_vllm True \
 --vllm_mode colocate