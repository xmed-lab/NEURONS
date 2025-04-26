from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="gongzx/cc2017_dataset",
    repo_type="dataset",
    local_dir="./cc2017_dataset",
    local_dir_use_symlinks=False
)

snapshot_download(
    repo_id="McGregorW/NEURONS",
    local_dir="./cc2017_dataset",
    local_dir_use_symlinks=False,
    allow_patterns=["masks/*", "qwen_annotation/*"],
)