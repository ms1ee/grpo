#!/usr/bin/env python3
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    snapshot_download(
        token="",
        repo_id="",
        repo_type="dataset",
        local_dir="data",
        cache_dir=".hf-cache",     # 필요 없다면 삭제 가능
        resume_download=True,
    )

