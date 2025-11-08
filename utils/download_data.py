"""Utility functions for downloading datasets and loading JSON files."""

import json
from pathlib import Path
from urllib import request
from typing import Any, Dict, Union
import os

def download_and_save(save_dir: Union[str, Path] = os.getenv("SAVE_DIR", "data")) -> None:
    """
    Download ``train.json`` and ``test.json`` from ``base_url`` into ``save_dir``.
    The directory is created if it does not exist. Files are only downloaded
    when they are missing.

    Args:
        save_dir: Destination directory path.
        base_url: Base URL.
    """
    base_url = "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main/dataset/hellaswag/"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for name in ["train.json", "test.json"]:
        out_file = save_dir / name
        if not out_file.exists():
            request.urlretrieve(base_url + name, out_file)

def load_json(dataset: str, 
              save_dir: Union[str, Path] = os.getenv("SAVE_DIR", "data")) -> Dict[str, Any]:
    """
    Ensure the dataset JSON file is present locally and load it.

    Args:
        dataset: Name of the dataset without extension (e.g. "train" or "test").
        save_dir: Directory where JSON files are stored.
        base_url: Base URL used to download the files if they are missing.

    Returns:
        Parsed JSON content as a dictionary.
    """
    download_and_save(save_dir)
    file_path = Path(save_dir) / f"{dataset}.json"
    with file_path.open("r", encoding="utf-8") as fid:
        return json.load(fid)

__all__ = ["download_and_save", "load_json"]