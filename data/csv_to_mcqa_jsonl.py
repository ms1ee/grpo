#!/usr/bin/env python3
import csv, json, argparse, random, os
from typing import List, Dict

CHOICE_KEYS = ["answer1","answer2","answer3","answer4","answer5"]
# 최대 개수
MAX_ROW = 3000

def normalize_label(v: str) -> int:
    """라벨을 0~4 인덱스로 통일. 입력은 '1'~'5' 또는 'A'~'E' 모두 허용."""
    s = str(v).strip().upper()
    if s in ["A","B","C","D","E"]:
        return ord(s) - ord("A")
    if s.isdigit():
        iv = int(s)
        if 1 <= iv <= 5: return iv -1
    else:
        print((f"Invalid solution value: {v} (expected 1~5 or A~E)"))
        pass
        # raise ValueError(f"Invalid solution value: {v} (expected 1~5 or A~E)")

def row_to_item(row: Dict[str,str]) -> Dict:
    # 컬럼 이름 공백/대소문자/BOM 제거
    norm = {k.strip().lstrip("\ufeff").lower(): v for k, v in row.items()}

    # 필수 컬럼 확인
    need = ["question", "solution"] + CHOICE_KEYS

    # 필수 컬럼 확인
    for k in need:
        if k not in norm:
            raise KeyError(f"Missing column: {k}. Got columns: {list(norm.keys())}")
    label_idx = normalize_label(row["solution"])
    item = {
        "question": row["question"],
        "answer1": row["answer1"],
        "answer2": row["answer2"],
        "answer3": row["answer3"],
        "answer4": row["answer4"],
        "answer5": row["answer5"],
        # 학습 스크립트가 자동 인식하도록 label을 1~5와 A~E 모두 저장(선택)
        "solution": label_idx + 1,                 # 1~5
        # "label_letter": chr(ord("A")+label_idx) # 'A'~'E'
    }
    return item

def write_jsonl(path: str, data: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_path", required=True, help="train jsonl path")
    ap.add_argument("--out_eval", default="", help="eval jsonl path (optional)")
    ap.add_argument("--val_ratio", type=float, default=0.0, help="0~1; >0이면 랜덤 분할")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    cnt = 0
    for inputfile in os.listdir(args.in_path):
        # if inputfile != "kmmlu_law.csv": continue
        if not inputfile.endswith("csv"): continue
        print(inputfile)
        with open(os.path.join(args.in_path, inputfile), "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                # if cnt >= MAX_ROW:
                #     break
                rows.append(row_to_item(r))
                cnt += 1

        random.seed(args.seed)
        random.shuffle(rows)

        outputfile = os.path.join(args.out_path, inputfile[:inputfile.find("csv")] + "jsonl")
        if args.val_ratio > 0 and outputfile:
            n_val = max(1, int(len(rows) * args.val_ratio))
            eval_set = rows[:n_val]
            train_set = rows[n_val:]
            write_jsonl(outputfile, train_set)
            write_jsonl(args.out_eval, eval_set)
            print(f"Saved {len(train_set)} train → {outputfile}")
            print(f"Saved {len(eval_set)}  eval  → {args.out_eval}")
        else:
            write_jsonl(outputfile, rows)
            if args.out_eval:
                print("val_ratio==0 이므로 eval 파일은 생성하지 않았습니다.")
            print(f"Saved {len(rows)} items → {outputfile}")

if __name__ == "__main__":
    main()

"""
python csv_to_mcqa_jsonl.py \
  --in_path data/ \
  --out_path data/jsonl \
  --out_eval data/eval.jsonl \
  --val_ratio 0
"""
