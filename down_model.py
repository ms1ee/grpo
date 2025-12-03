#!/usr/bin/env python3
import argparse
from huggingface_hub import snapshot_download

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="ex) username/my-model")
    p.add_argument("--revision", default=None, help="branch/tag/commit (optional)")
    p.add_argument("--local-dir", default="./hf_download", help="download dir")
    p.add_argument("--include", nargs="*", default=None,
                   help="allow_patterns (ex: *.safetensors adapter_config.json)")
    p.add_argument("--exclude", nargs="*", default=None,
                   help="ignore_patterns (ex: *.bin *.pt)")
    p.add_argument("--token", default=None, help="HF token (optional; or use env)")
    args = p.parse_args()

    path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,   # 실파일 복사(컨테이너/압축 이동 편함)
        allow_patterns=args.include,    # 포함 패턴
        ignore_patterns=args.exclude,   # 제외 패턴
        token=args.token,               # 또는 HUGGINGFACE_HUB_TOKEN 환경변수 사용
        resume_download=True,
        force_download=True,
        )
    print(f"✅ Downloaded to: {path}")

if __name__ == "__main__":
    main()


