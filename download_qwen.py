import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import snapshot_download
import sys

print("Starting Qwen3-14B download...")
try:
    path = snapshot_download(
        repo_id="Qwen/Qwen3-14B", 
        local_dir="/mnt/projects1/daedalus/models/lineage/base_v000.bf16", 
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"Download completed successfully. Path: {path}")
except Exception as e:
    print(f"Download failed: {e}")
    sys.exit(1)
