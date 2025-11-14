import re
from typing import List, Dict, Any

CHOICE_KEYS = ["answer1","answer2","answer3","answer4","answer5"]
NUM_RE = re.compile(r"\b([1-5])\b")
TAG_NUM_RE = re.compile(r"<\s*Answer\s*>\s*([1-5])\s*<\s*/\s*Answer\s*>", re.I)


def build_prompt(q: str, choices: List[str], tok) -> str:
    system = {
        "role": "system",
        "content": (
            "You are a precise multiple-choice solver.\n"
            "Write a brief multi-line reasoning trace first (at least 6 separate lines).\n"
            "Then, on the LAST line ONLY, output exactly: <Answer> k </Answer> (k is 1..5).\n"
            "Do not write anything after the closing tag </Answer>.\n"
            "Do not restate options in the final line. Stop immediately after printing the tag."
        ),
    }

    # --- Few-shot 1: 최소 6줄 이상의 추론 + 마지막 줄에만 정답 태그 ---
    shot1_user = {
        "role": "user",
        "content":
            "Question:\nA or B?\n\n"
            "Options:\n1. A\n2. B\n3. C\n4. D\n5. E\n\n"
            "Return ONLY the final choice in the last line as <Answer> k </Answer>."
    }
    shot1_assist = {
        "role": "assistant",
        "content": (
            "Identify what the question asks.\n"
            "Compare A vs B directly.\n"
            "Check any constraints given.\n"
            "Eliminate irrelevant choices (C, D, E).\n"
            "Confirm the better fit among A and B.\n"
            "Therefore, the correct option is 1.\n"
            "<Answer> 1 </Answer>"
        )
    }

    # --- Few-shot 2 ---
    shot2_user = {
        "role": "user",
        "content":
            "Question:\nPick D.\n\n"
            "Options:\n1. A\n2. B\n3. C\n4. D\n5. E\n\n"
            "Return ONLY the final choice in the last line as <Answer> k </Answer>."
    }
    shot2_assist = {
        "role": "assistant",
        "content": (
            "Parse the instruction.\n"
            "Target option is D.\n"
            "Map D to its index.\n"
            "Double-check numbering (1..5).\n"
            "Ensure the final line uses the exact tag format.\n"
            "No text after the tag is allowed.\n"
            "<Answer> 4 </Answer>"
        )
    }

    opts_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
    user = {
        "role": "user",
        "content": (
            f"Question:\n{q}\n\n"
            f"Options:\n{opts_text}\n\n"
            "Write a brief reasoning trace across multiple lines (≥6 lines).\n"
            "Then, on the LAST line ONLY, output exactly: <Answer> k </Answer>.\n"
            "Do not add anything after the tag."
        ),
    }

    messages = [system, shot1_user, shot1_assist, shot2_user, shot2_assist, user]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_final_answer(gen: str):
    pred, has_tag, end_pos = "", False, -1
    last = None
    for m in TAG_NUM_RE.finditer(gen):
        last = m
    if last is not None:
        pred = last.group(1)
        has_tag = True
        end_pos = last.end()
    return pred, has_tag, end_pos


def detect_label(example: Dict[str, Any]) -> int:
    if "label" in example:
        idx = int(example["label"]) - 1
        if idx not in range(5): raise ValueError("label must be 1..5")
        return idx
    if "label_letter" in example:
        return ord(str(example["label_letter"]).strip().upper()) - ord("A")
    if "solution" in example:
        s = str(example["solution"]).strip().upper()
        if s in ["A","B","C","D","E"]:
            return ord(s) - ord("A")
        if s.isdigit():
            return int(s) - 1
    raise ValueError("No label found in example.")
