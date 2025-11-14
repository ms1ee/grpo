from typing import Tuple
import torch, os, requests
from transformers import TrainerCallback, GenerationConfig
from .env import IS_MAIN

try:
    import wandb
    _WANDB = True
except Exception:
    _WANDB = False

from .prompt import parse_final_answer

# === vLLM(OpenAI 호환) 헬퍼 ===
def _vllm_completion(prompt: str, *, base_url: str, api_key: str, model: str,
                     max_new_tokens: int = 2048, temperature: float = 0.0, top_p: float = 1.0,
                     timeout: float = 60.0) -> str:
    """vLLM OpenAI /v1/completions 엔드포인트 호출 (greedy 기본)."""
    url = base_url.rstrip("/") + "/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 2048,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    # vLLM OpenAI 호환 응답 파싱
    return data["choices"][0].get("text", "")


# -------------------------------
# helpers for pretty printing / logging
# -------------------------------
def _preview(s: str, head: int = 160, tail: int = 160) -> str:
    s = s.replace("\n", "\\n")
    if len(s) <= head + tail:
        return s
    return f"{s[:head]} … {s[-tail:]}"


@torch.no_grad()
def evaluate_accuracy(
    model,
    tok,
    eval_ds,
    max_samples: int = 512,
    max_new_tokens: int = 2048,
    device: str = "cuda",
    # ---- NEW: backend 선택 ----
    backend: str = "vllm",   # "local" | "vllm"
    openai_base: str | None = None,
    openai_key:  str | None = None,
    openai_model: str | None = None,
    # ---- NEW: debug / logging options ----
    log_samples: bool = True,
    log_n: int = 3,
    preview_chars: int = 160,
    wandb_table: bool = False,
) -> Tuple[float, int]:
    """
    마지막 <Answer> 태그에서 추출한 선택지를 gold와 비교해 정확도를 계산한다.
    backend="vllm" 이면 외부 vLLM(OpenAI 호환)으로 생성한다.
    """
    eval_gcfg = GenerationConfig(
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    if IS_MAIN:
        print("[debug] EVAL GenerationConfig:", eval_gcfg.to_dict())
        first3 = [int(eval_ds[i]["label_idx"]) + 1 for i in range(min(3, len(eval_ds)))]
        print("[debug] eval first 3 gold labels:", first3)
        print(f"[debug] eval backend={backend}")

    model_was_training = model.training
    model.eval()
    total_targets = min(len(eval_ds), max_samples)
    correct = 0
    evaluated = 0

    # (선택) W&B 테이블 초기화
    wb_table = None
    if _WANDB and wandb_table and IS_MAIN:
        wb_table = wandb.Table(
            columns=[
                "idx", "gold", "pred", "matched",
                "prompt_preview", "gen_preview",
            ]
        )

    # vLLM 연결정보 자동 폴백
    if backend == "vllm":
        openai_base  = openai_base or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8010/v1")
        openai_key   = openai_key  or os.environ.get("OPENAI_API_KEY", "EMPTY")
        if openai_model is None:
            # 토크나이저 폴더에 model 정보를 안 넣었으면 env로 받음(없어도 vLLM은 무시 가능)
            openai_model = os.environ.get("OPENAI_MODEL_NAME", "unknown")

    for i in range(total_targets):
        e = eval_ds[i]
        prompt = e["prompt"]
        gold   = str(int(e["label_idx"]) + 1)

        # --- 생성 ---
        if backend == "vllm":
            try:
                gen = _vllm_completion(
                    prompt,
                    base_url=openai_base,
                    api_key=openai_key,
                    model=openai_model,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0, top_p=1.0, timeout=60.0,
                )
            except Exception as ex:
                if IS_MAIN:
                    print(f"[eval vLLM error @ {i}]: {ex}")
                # 실패 시 스킵
                continue
        else:
            # 기존 로컬 경로
            inputs = tok(prompt, return_tensors="pt").to(device)
            try:
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=False,                      # ✅ 로컬 평가 시 메모리 폭증 방지
                    generation_config=eval_gcfg,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id,
                )
                text = tok.decode(out[0], skip_special_tokens=True)
                gen  = text[len(prompt):] if text.startswith(prompt) else text
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue

        pred, has_tag, _ = parse_final_answer(gen)
        evaluated += 1
        matched = has_tag and (pred == gold)
        if matched:
            correct += 1

        # (선택) 콘솔에 몇 개만 자세히 찍기
        if IS_MAIN and log_samples and i < log_n:
            print("\n[eval-sample]", i)
            print("  gold:", gold, " pred:", repr(pred), " matched:", matched, " has_tag:", has_tag)
            print("  prompt_preview:", _preview(prompt, preview_chars, preview_chars))
            print("  gen_preview:   ", _preview(gen,    preview_chars, preview_chars))

        if wb_table is not None:
            wb_table.add_data(
                i, gold, pred, matched,
                _preview(prompt, preview_chars, preview_chars),
                _preview(gen,    preview_chars, preview_chars),
            )

    if _WANDB and wb_table is not None and IS_MAIN:
        wandb.log({"eval/samples_table": wb_table})

    acc = correct / evaluated if evaluated > 0 else 0.0
    if model_was_training:
        model.train()
    return acc, evaluated


class PeriodicEvalCallback(TrainerCallback):
    def __init__(self, tok, eval_ds, every, max_samples,
                 device="cuda",
                 # ---- NEW ----
                 backend: str = "local",          # "local" | "vllm"
                 openai_base: str | None = None,
                 openai_key:  str | None = None,
                 openai_model: str | None = None,
                 # pass-through options
                 log_samples: bool = True,
                 log_n: int = 3,
                 preview_chars: int = 160,
                 wandb_table: bool = True):
        self.tok = tok
        self.eval_ds = eval_ds
        self.every = max(1, every)
        self.max_samples = max_samples
        self.device = device
        self.trainer = None
        self._last_eval_step = -1

        self.backend = backend
        self.openai_base = openai_base
        self.openai_key = openai_key
        self.openai_model = openai_model

        self._log_samples = log_samples
        self._log_n = log_n
        self._preview_chars = preview_chars
        self._wandb_table = wandb_table

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer = kwargs.get("trainer", None)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not IS_MAIN or self.eval_ds is None:
            return control

        step = state.global_step
        if step <= 0 or step == self._last_eval_step or (step % self.every) != 0:
            return control

        model = self.trainer.model if self.trainer is not None else kwargs.get("model", None)
        if model is None:
            return control

        completion_len = getattr(args, "max_completion_length", self._max_completion_fallback)
        acc, n = evaluate_accuracy(
            model, self.tok, self.eval_ds,
            self.max_samples,
            max_new_tokens=completion_len,
            device=self.device,
            backend=self.backend,
            openai_base=self.openai_base,
            openai_key=self.openai_key,
            openai_model=self.openai_model,
            log_samples=self._log_samples,
            log_n=self._log_n,
            preview_chars=self._preview_chars,
            wandb_table=self._wandb_table,
        )
        print(f"[Eval @ step {step}] acc={acc:.4f} n={n}")

        if _WANDB:
            wandb.log({"eval/accuracy": acc, "eval/step": step, "eval/num_samples": n})

        self._last_eval_step = step
        return control


class ConsoleLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not IS_MAIN:
            return
        metrics = {k: v for k, v in logs.items() if k != "epoch"}
        s = " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float)))
        print(f"[Step {state.global_step}] {s}")
