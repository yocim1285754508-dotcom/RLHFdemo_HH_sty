from pathlib import Path

from huggingface_hub import snapshot_download

from config import Config


MODEL_CACHE_ROOT = Config.PROJECT_ROOT / ".hf_models"
MODEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def _safe_model_dir_name(model_name: str) -> str:
    return model_name.replace("/", "--")


def resolve_model_path(model_name_or_path: str) -> str:
    local_path = Path(model_name_or_path)
    if local_path.exists():
        return str(local_path)

    target_dir = MODEL_CACHE_ROOT / _safe_model_dir_name(model_name_or_path)
    if target_dir.exists() and any(target_dir.iterdir()):
        return str(target_dir)

    snapshot_download(
        repo_id=model_name_or_path,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    return str(target_dir)
