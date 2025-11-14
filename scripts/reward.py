from typing import List
from .prompt import parse_final_answer
from .env import IS_MAIN, GLOBAL_STEP_FOR_LOG

try:
    import wandb  # not used inside reward, but keep symmetry if needed
    _WANDB = True
except Exception:
    _WANDB = False


_WB_TABLE = None  # persistent table (메인 프로세스에서만 사용)

def make_mcqa_reward(cfg):
    def mcqa_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        labels: List[int] = kwargs["label_idx"]
        gold_nums = [str(i + 1) for i in labels]

        fmt_ok = newlines = correct_ok = 0
        sum_format = sum_trace = sum_correct = 0.0

        rows = []
        scores: List[float] = []
        example_to_log = None
        for idx, (p, comp, gold) in enumerate(zip(prompts, completions, gold_nums)):
            gen = comp[len(p):] if comp.startswith(p) else comp
            pred, has_tag, _ = parse_final_answer(gen)

            s_format  = 1.0 if has_tag and pred in {"1","2","3","4","5"} else 0.0
            s_trace   = 1.0 if gen.count('\n') >= 6 else 0.0
            s_correct = 1.0 if has_tag and pred == gold else 0.0

            score = (cfg.w_format * s_format) + (cfg.w_trace * s_trace) + (cfg.w_correct * s_correct)
            scores.append(float(score))

            fmt_ok     += int(s_format == 1.0)
            newlines   += int(s_trace  == 1.0)
            correct_ok += int(s_correct == 1.0)
            sum_format  += float(s_format)
            sum_trace   += float(s_trace)
            sum_correct += float(s_correct)

            if example_to_log is None:
                example_to_log = {
                    "idx": idx,
                    "prompt": p,
                    "generation": gen,
                    "score": score,
                    "s_format": s_format,
                    "s_trace": s_trace,
                    "s_correct": s_correct,
                    "has_tag": has_tag,
                    "pred": pred or "",
                    "gold": gold,
                }

            rows.append([
                int(idx),
                str(gold),
                "" if pred is None else str(pred),
                int(bool(has_tag)),
                float(score),
                float(s_format),
                float(s_trace),
                float(s_correct),
                p, # pr_preview,
                gen, # gen_preview,
            ])

        N = max(1, len(completions))
        mean_score   = float(sum(scores) / N)
        mean_format  = float(sum_format  / N)
        mean_trace   = float(sum_trace   / N)
        mean_correct = float(sum_correct / N)

        if IS_MAIN and (GLOBAL_STEP_FOR_LOG % cfg.log_samples_every == 0):
            print(
                f"[reward-stats] mean={mean_score:.4f} "
                f"fmt_ok={fmt_ok}/{N} trace_ok={newlines}/{N} correct_ok={correct_ok}/{N} "
                f"(format={mean_format:.3f}, trace={mean_trace:.3f}, correct={mean_correct:.3f})"
            )
            if example_to_log is not None:
                print("[reward-sample] idx={idx} gold={gold} pred={pred} score={score:.4f} "
                        "format={s_format:.1f} trace={s_trace:.1f} correct={s_correct:.1f}".format(**example_to_log))
                print("  prompt:", example_to_log["prompt"])
                print("  generation:", example_to_log["generation"])

        # 스텝 증가
        from . import env
        env.GLOBAL_STEP_FOR_LOG += 1
        step = env.GLOBAL_STEP_FOR_LOG

        # ✅ W&B 로깅
        if _WANDB and IS_MAIN:
            # 1) 요약 스칼라/히스토그램 (주기적으로)
            if cfg.log_samples_every > 0 and (step % cfg.log_samples_every == 0):
                wandb.log({
                    "reward/mean_total":   mean_score,
                    "reward/format_mean":  mean_format,
                    "reward/trace_mean":   mean_trace,
                    "reward/correct_mean": mean_correct,
                    "reward/format_ratio":  float(fmt_ok / N),
                    "reward/trace_ratio":   float(newlines / N),
                    "reward/correct_ratio": float(correct_ok / N),
                    "reward/batch_size":    int(N),
                    "reward/scores_hist":   wandb.Histogram(scores),
                })

            # 2) per-sample 테이블 (지속 인스턴스에 append)
            global _WB_TABLE
            if _WB_TABLE is None:
                _WB_TABLE = wandb.Table(
                    columns=[
                        "idx","gold","pred","has_tag",
                        "score","s_format","s_trace","s_correct",
                        "prompt","gen"
                    ],
                    log_mode="INCREMENTAL",
                )

            # rows가 비어도 add_data는 생략
            for r in rows:
                _WB_TABLE.add_data(*r)

            # 테이블은 너무 자주 갱신하면 느려질 수 있으니, 같은 주기로만 푸시
            if cfg.log_samples_every > 0 and (step % 10 == 0):
                wandb.log({"reward/samples_table": _WB_TABLE})

        return scores
    return mcqa_reward
