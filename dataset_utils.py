from pathlib import Path

import requests
from datasets import load_dataset

from config import Config


PKU_DATASET_ROOT = Config.PROJECT_ROOT / ".pku_saferlhf"
PKU_DATASET_ROOT.mkdir(parents=True, exist_ok=True)
HH_DATASET_ROOT = Config.PROJECT_ROOT / ".anthropic_hh"
HH_DATASET_ROOT.mkdir(parents=True, exist_ok=True)

PKU_FILES = {
    "train": [
        ("Alpaca-7B", "train.jsonl"),
        ("Alpaca2-7B", "train.jsonl"),
        ("Alpaca3-8B", "train.jsonl"),
    ],
    "test": [
        ("Alpaca-7B", "test.jsonl"),
        ("Alpaca2-7B", "test.jsonl"),
        ("Alpaca3-8B", "test.jsonl"),
    ],
}

HH_FILES = {
    "train": [
        ("helpful-base", "train.jsonl.gz"),
    ],
    "test": [
        ("helpful-base", "test.jsonl.gz"),
    ],
}


def _pku_url(subdir: str, filename: str) -> str:
    return (
        "https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF/resolve/main/"
        f"data/{subdir}/{filename}"
    )


def _download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp_path.replace(dest)


def ensure_pku_local_files():
    local_files = {"train": [], "test": []}
    for split, file_specs in PKU_FILES.items():
        for subdir, filename in file_specs:
            local_path = PKU_DATASET_ROOT / subdir / filename
            if not local_path.exists():
                print(f"[Dataset] Downloading {subdir}/{filename} ...")
                _download_file(_pku_url(subdir, filename), local_path)
            local_files[split].append(str(local_path))
    return local_files


def load_local_pku_dataset():
    data_files = ensure_pku_local_files()
    return load_dataset("json", data_files=data_files)


def _hh_url(subdir: str, filename: str) -> str:
    return (
        "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/"
        f"{subdir}/{filename}"
    )


def ensure_hh_local_files():
    local_files = {"train": [], "test": []}
    for split, file_specs in HH_FILES.items():
        for subdir, filename in file_specs:
            local_path = HH_DATASET_ROOT / subdir / filename
            if not local_path.exists():
                print(f"[Dataset] Downloading HH {subdir}/{filename} ...")
                _download_file(_hh_url(subdir, filename), local_path)
            local_files[split].append(str(local_path))
    return local_files


def load_local_hh_dataset():
    data_files = ensure_hh_local_files()
    return load_dataset("json", data_files=data_files)
