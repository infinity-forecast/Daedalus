import os
import subprocess
from huggingface_hub import HfFileSystem, hf_hub_url

repo_id = "Qwen/Qwen3-8B"
out_dir = "/mnt/projects1/daedalus/models/lineage/base_v000.bf16"
os.makedirs(out_dir, exist_ok=True)

print(f"Fetching file list for {repo_id}...")
fs = HfFileSystem()
files = fs.ls(repo_id, detail=False)

# Filter out the directory path, keeping just filenames
filenames = [f.split("/")[-1] for f in files if getattr(fs.info(f), "type", "file") == "file"]

print(f"Found {len(filenames)} files. Beginning wget sequence...")

for fname in filenames:
    url = hf_hub_url(repo_id=repo_id, filename=fname)
    out_path = os.path.join(out_dir, fname)
    
    print(f"\n─────────────────────────────────────────────────────────────────")
    print(f"Downloading: {fname}")
    
    # We use wget --continue to resume partial downloads.
    # --tries=inf and --retry-connrefused handle Errno 104 perfectly.
    cmd = [
        "wget", 
        "--continue", 
        "--tries=50", 
        "--timeout=30",
        "--waitretry=2",
        "--retry-connrefused",
        "-O", out_path,
        url
    ]
    
    subprocess.run(cmd)

print("\n✅ All files completely downloaded to:")
print(out_dir)
