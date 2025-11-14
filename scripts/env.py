import os

RANK = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0") or 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
IS_MAIN = (RANK == 0)

# reward에서 쓰던 글로벌 스텝 카운터 (샘플 디버그 출력용)
GLOBAL_STEP_FOR_LOG = 0