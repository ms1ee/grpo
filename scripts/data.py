from datasets import load_dataset
from typing import Dict, Any
from .prompt import CHOICE_KEYS, build_prompt, detect_label
from .env import IS_MAIN
import builtins

def map_example_factory(tok, cfg):
    def map_example(e: Dict[str, Any]) -> Dict[str, Any]:
        q = e["question"]
        choices = [e[k] for k in CHOICE_KEYS]
        prompt = build_prompt(q, choices, tok)
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) > cfg.max_completion_length:
            prompt = tok.decode(ids[-cfg.max_completion_length:], skip_special_tokens=True)
        label_idx = detect_label(e)

        return {"prompt": prompt, "label_idx": label_idx}
    return map_example


def load_train_eval(cfg, tok):
    # train
    if cfg.dataset_name in ["json","jsonl"]:
        ds = load_dataset(cfg.dataset_name, data_files=cfg.data_files, split=cfg.split)
    else:
        ds = load_dataset(cfg.dataset_name, split=cfg.split)

    map_example = map_example_factory(tok, cfg)
    train_ds = ds.map(map_example, remove_columns=[c for c in ds.column_names if c not in {"prompt","label_idx"}])

    # eval (optional)
    eval_ds = None
    if cfg.eval_data_files:
        if cfg.dataset_name in ["json","jsonl"]:
            eds = load_dataset(cfg.dataset_name, data_files=cfg.eval_data_files, split="train")
        else:
            eds = load_dataset(cfg.dataset_name, data_files=cfg.eval_data_files, split="train")
        eval_ds = eds.map(map_example, remove_columns=[c for c in eds.column_names if c not in {"prompt","label_idx"}])

    return train_ds, eval_ds
