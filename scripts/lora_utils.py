import torch
from peft import LoraConfig, TaskType, PeftModel


def cast_all_lora_to_dtype(model, dtype=torch.bfloat16):
    for n, p in model.named_parameters():
        if "lora_" in n and p.dtype != dtype:
            p.data = p.data.to(dtype)


def make_lora_config_from_cfg(cfg):
    tmods = [s.strip() for s in cfg.rl_target_modules.split(",") if s.strip()]
    return LoraConfig(
        r=cfg.rl_r,
        lora_alpha=cfg.rl_alpha,
        lora_dropout=cfg.rl_dropout,
        modules_to_save=None,  # ensure no dense weights are persisted for vLLM compatibility
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=tmods,
    )


def freeze_all_params(model):
    for p in model.parameters():
        p.requires_grad_(False)


def make_only_lora_trainable(model, adapter_name: str | None = None):
    freeze_all_params(model)
    token = f".{adapter_name}." if adapter_name else None
    for n, p in model.named_parameters():
        if "lora_" in n:
            if token is None or (token in n):
                p.requires_grad_(True)
    return model


def try_set_adapter(model, adapters):
    try:
        model.set_adapter(adapters)
    except Exception:
        if isinstance(adapters, (list, tuple)) and len(adapters) > 0:
            model.set_adapter(adapters[-1])
