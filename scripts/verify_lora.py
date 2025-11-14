sft_lora_dir = "../qwen3-sft/checkpoints/qwen3_sft_lora_bf16/checkpoint-626"
from peft import LoraConfig  # just to ensure dependency is present

if cfg.rl_mode.lower() == "continue":
    model = PeftModel.from_pretrained(model, sft_lora_dir, is_trainable=True)
    active = getattr(model, "active_adapter", None) or "default"
    try_set_adapter(model, active)
    make_only_lora_trainable(model, adapter_name=active)
else:
    model = PeftModel.from_pretrained(model, sft_lora_dir, adapter_name="sft", is_trainable=False)
    grpo_cfg = make_lora_config_from_cfg(cfg)
    model.add_adapter("grpo", grpo_cfg)
    try_set_adapter(model, ["sft","grpo"])
    make_only_lora_trainable(model, adapter_name="grpo")

# === [VERIFY A] LoRA fingerprint 동일성 확인 ===
import hashlib, gc

def lora_fingerprint(m):
    h = hashlib.sha256()
    sd = m.state_dict()
    for k in sorted(sd.keys()):
        if "lora_" in k:
            t = sd[k].detach().float().cpu().contiguous()
            h.update(t.numpy().tobytes())
    return h.hexdigest()

# 현재 학습에 사용할 모델의 LoRA 지문
fp_curr = lora_fingerprint(model)

# 참조(SFT LoRA) 모델을 따로 로드해서 지문 생성
base_ref = AutoModelForCausalLM.from_pretrained(
    cfg.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
)
ref = PeftModel.from_pretrained(base_ref, sft_lora_dir, is_trainable=False)
fp_ref = lora_fingerprint(ref)

print(f"[verify] lora_fingerprint current = {fp_curr}")
print(f"[verify] lora_fingerprint   sft   = {fp_ref}")
print("[verify] SAME? ", fp_curr == fp_ref)

# 메모리 회수
del ref, base_ref; gc.collect(); torch.cuda.empty_cache()
# === Verify 끝 ===