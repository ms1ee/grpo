#!/usr/bin/env python
from __future__ import annotations
import argparse, json, logging, os, torch, wandb
from dataclasses import asdict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from peft import PeftModel, prepare_model_for_kbit_training, get_peft_model
from .config import ScriptArgs
from .env import IS_MAIN
from .data import load_train_eval
from .reward import make_mcqa_reward
from .eval_utils import PeriodicEvalCallback, ConsoleLoggerCallback
from .lora_utils import *

# in 'colocate' mode, below those are not used.
MAX_NEW_TOKENS = 2048
TENSOR_PARALLEL_SIZE = 2
GPU_MEM_UTIL = 0.7

def main():
    parser = argparse.ArgumentParser()
    for k, v in ScriptArgs.__dataclass_fields__.items():
        parser.add_argument(f"--{k}", type=type(v.default), default=v.default)
    args, _ = parser.parse_known_args()
    cfg = ScriptArgs(**vars(args))

    env_max_new = os.environ.get("MAX_NEW_TOKENS")
    if env_max_new:
        try:
            cfg.max_completion_length = int(env_max_new)
        except ValueError:
            logging.warning("Invalid MAX_NEW_TOKENS=%s; keeping CLI/config value %d",
                            env_max_new, cfg.max_completion_length)

    # ---- wandb
    if not IS_MAIN:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DISABLED"] = "true"
    if IS_MAIN:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=asdict(cfg),
            settings=wandb.Settings(start_method="thread", _service_wait=30),
            reinit=True
        )
    report_to = ["tensorboard", "wandb"] if IS_MAIN else ["tensorboard"]

    # ---- logging & seed
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(level=logging.INFO)
    logging.info("CONFIG\n%s", json.dumps(asdict(cfg), indent=2))
    torch.manual_seed(cfg.seed)

    # ---- tokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    tok.padding_side = "left"
    added_pad_token = False
    if (tok.pad_token is None) or (tok.pad_token_id == tok.eos_token_id):
        if cfg.use_vllm:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
            added_pad_token = True

    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.sft_full_checkpoint if cfg.rl_mode.lower() == "sft-full" else cfg.model_name,
        quantization_config=None,
        device_map=None,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=True
    )
    model = prepare_model_for_kbit_training(model)
    if added_pad_token:
        model.resize_token_embeddings(len(tok))

    # sft의 checkpoint 이어서 할지, vanilla 위에서 할지.
    from peft import LoraConfig  # just to ensure dependency is present

    if cfg.rl_mode.lower() == "sft-lora":
        ## Load SFT LoRA addapter
        model = PeftModel.from_pretrained(model, cfg.sft_lora_checkpoint, is_trainable=True)
        active = getattr(model, "active_adapter", None) or "default"
        try_set_adapter(model, active)
        make_only_lora_trainable(model, adapter_name=active)
    elif cfg.rl_mode.lower() == "sft-full":
        ## Load SFT Full Checkpoint
        for p in model.parameters():
            p.requires_grad_(True)
    elif cfg.rl_mode.lower() == "baseline":
        grpo_cfg = make_lora_config_from_cfg(cfg)
        model = get_peft_model(model, grpo_cfg)
        try_set_adapter(model, "default")
        make_only_lora_trainable(model)
    else:
        raise ValueError(f"Unknown rl_mode {cfg.rl_mode}")

    cast_all_lora_to_dtype(model, torch.bfloat16)

    def _align_to_head_dtype(module, inputs):
        (hidden_states,) = inputs
        tgt = module.weight.dtype
        if hidden_states.dtype != tgt:
            hidden_states = hidden_states.to(tgt)
        return (hidden_states,)
    _ = model.get_output_embeddings().register_forward_pre_hook(_align_to_head_dtype)

    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False

    model.train()

    # Data loading
    # Currently, for the saving training time, do not conduct evaluation
    train_ds, eval_ds = load_train_eval(cfg, tok)
    if cfg.use_vllm:
        gen_kwargs = {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "repetition_penalty": 1.07,
            "max_tokens": cfg.max_completion_length,   # vLLM은 max_tokens 사용
            "stop_token_ids": [im_end_id],             # eos 대용
        }
    else:
        gen_kwargs = {
            "use_cache": True,
            "do_sample": True,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "repetition_penalty": 1.07,
            "max_new_tokens": cfg.max_completion_length,
            "eos_token_id": im_end_id,
            "pad_token_id": tok.pad_token_id,
            "num_beams": 1,
        }
    lr_scheduler_kwargs = {
        "min_lr_rate": 0.1,
    }
    # ---- GRPO args
    grpo = GRPOConfig(
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        num_generations=cfg.num_generations,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        beta=0.1,
        loss_type=cfg.loss_type,
        learning_rate=cfg.lr,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        per_device_train_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        bf16=True,
        logging_strategy="steps",
        logging_steps=cfg.log_samples_every,
        log_level="info",
        report_to=report_to,
        output_dir=cfg.output_dir,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        ddp_find_unused_parameters=cfg.ddp_find_unused_parameters,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=cfg.dataloader_pin_memory,
        dataloader_drop_last=cfg.dataloader_drop_last,
        use_vllm=cfg.use_vllm,
        vllm_mode=cfg.vllm_mode,
        generation_kwargs=gen_kwargs,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        save_total_limit=cfg.save_total_limit,
    )

    # optimizer
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )

    # trainer
    reward_fn = make_mcqa_reward(cfg)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo,
        train_dataset=train_ds,
        processing_class=tok,
        optimizers=(optim, None),
    )

    trainer.model.config.use_cache = False
    trainer.train()
    trainer.save_model(cfg.output_dir)

    print(f"GRPO train end! checkpoint is saved into '{cfg.output_dir}'")

if __name__ == "__main__":
    main()
