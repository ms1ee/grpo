from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ScriptArgs:
    model_name: str = "Qwen/Qwen3-1.7B"
    dataset_name: str = "json"
    data_files: str = "data/train.jsonl"
    split: str = "train"
    eval_data_files: str = "data/eval.json"
    output_dir: str = ""
    sft_full_checkpoint: str = ""
    sft_lora_checkpoint: str = ""
    resume_from_checkpoint: str = ""

    max_prompt_length: int = 2048
    max_completion_length: int = 3072

    temperature: float = 0.7
    top_p: float = 0.95
    lr: float = 5e-6
    lr_scheduler_type: str = "cosine_with_min_lr"
    loss_type: str = "dr_grpo"
    num_epochs: float = 3.0
    max_steps: int = -1
    per_device_batch: int = 1
    grad_accum: int = 4
    seed: int = 42
    head_dtype: str = "bf16"

    # reward 가중치
    w_format: float = 0.3
    w_trace: float = 0.2
    w_correct: float = 0.5

    num_generations: int = 8
    warmup_ratio: float = 0.1

    # eval & wandb
    eval_every: int = 100
    eval_max_samples: int = 512
    wandb_project: str = ""
    wandb_run_name: str = ""

    # RL & LoRA
    rl_mode: str = "sft-full"  # "sft-full" | "sft-lora" | "baseline"
    rl_r: int = 16
    rl_alpha: int = 16
    rl_dropout: float = 0.05
    rl_target_modules: str = "q_proj,k_proj,v_proj,o_proj,w1,w2,w3"

    # logging
    log_samples_every: int = 10
    log_n_samples: int = 1
    log_sample_max_chars: int = 500

    # Saving
    save_strategy: str = "epoch"
    save_steps: int = 1000
    save_total_limit: int = 3
    use_vllm: bool = True
    vllm_mode: str = "colocate"

    # ddp & dataloader
    ddp_find_unused_parameters: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    dataloader_drop_last: bool = False