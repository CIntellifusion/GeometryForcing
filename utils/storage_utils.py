"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

import torch
from pathlib import Path


def safe_torch_save(obj, path: Path):
    """
    Safely save a torch object to disk.
    If the path does not exist, create folders as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)
