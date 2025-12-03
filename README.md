# GRPO

GRPO Training with Qwen3-1.7B

---

## 디렉토리 구성

```bash
train/grpo
├── scripts/                  # GRPO 학습 관련 스크립트
│   ├── train_grpo.py         # GRPO 학습 메인 코드
│   ├── config.py             # GRPO 하이퍼파라미터 설정
│   ├── data.py               # 학습/평가 데이터 로딩 및 전처리
│   ├── prompt.py
│   ├── reward.py             # MCQA 기반 reward function 정의
│   ├── eval_utils.py         # 주기적 evaluation을 위한 함수
│   └── lora_utils.py         # LoRA 관련 유틸 함수
├── data/                     # 학습/평가 데이터 저장 디렉토리
│   ├── legal_ko.jsonl        # 한글 법률 데이터셋
│   ├── kmmlu_law.jsonl       # 평가 데이터
│   ├── train.jsonl           # Full 데이터셋
│   ├── csv_to_mcqa_jsonl.py
│   ├── random_select.py
│   └── token_stats_grpo.py
├── sft-model/                # SFT Full trained 모델
├── checkpoints/              # GRPO 학습된 모델 weight
├── utils/                    # HF 관련 유틸
│   ├── push_model.py
│   ├── push_dataset.py
│   └── pull_dataset.py
├── wandb/                    # W&B Log
├── run_grpo.sh               # GRPO 학습 실행 스크립트
├── grpo_environment.yaml     # conda 환경 설정
├── Dockerfile
├── build.sh                  # Docker 이미지 빌드
├── launch.sh                 # Docker 컨테이너 실행
├── down_model.py             # HF 모델 다운로드
└── down_model.sh             # 모델 다운로드 스크립트
```
<br>

## 환경 준비

```bash
conda env create -f grpo_environment.yaml
conda activate grpo
```
<br>

## 학습 실행
- GPU 4장 기준
- 학습 데이터: `data/legal_ko.jsonl`
```bash
./run_grpo.sh
```