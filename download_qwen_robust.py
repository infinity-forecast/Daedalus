import os
import sys
import time
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
repo_id = "Qwen/Qwen3-8B"
local_dir = "/mnt/projects1/daedalus/models/lineage/base_v000.bf16"

print(f"Starting robust download of {repo_id}...", flush=True)

while True:
    try:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"\nDownload completed successfully. Path: {path}", flush=True)
        break  # success!
    except BaseException as e:
        print(f"\nConnection or download error: {e}", flush=True)
        print("Retrying in 5 seconds...", flush=True)
        time.sleep(5)
