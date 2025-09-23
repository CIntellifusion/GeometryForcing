from typing import Any, Dict, Optional
import io
import tarfile
import torch
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from torchvision import transforms
from .base_video import (
    BaseVideoDataset,
    BaseSimpleVideoDataset,
    BaseAdvancedVideoDataset,
)


class MinecraftBaseVideoDataset(BaseVideoDataset):
    _ALL_SPLITS = ["training", "validation"]

    def download_dataset(self):
        from internetarchive import download

        part_suffixes = [
            "aa",
            "ab",
            "ac",
            "ad",
            "ae",
            "af",
            "ag",
            "ah",
            "ai",
            "aj",
            "ak",
        ]
        for part_suffix in part_suffixes:
            identifier = f"minecraft_marsh_dataset_{part_suffix}"
            file_name = f"minecraft.tar.part{part_suffix}"
            download(identifier, file_name, destdir=self.save_dir, verbose=True)

        combined_bytes = io.BytesIO()
        for part_suffix in part_suffixes:
            identifier = f"minecraft_marsh_dataset_{part_suffix}"
            file_name = f"minecraft.tar.part{part_suffix}"
            part_file = self.save_dir / identifier / file_name
            with open(part_file, "rb") as part:
                combined_bytes.write(part.read())
        combined_bytes.seek(0)
        with tarfile.open(fileobj=combined_bytes, mode="r") as combined_archive:
            combined_archive.extractall(self.save_dir)
        (self.save_dir / "minecraft/test").rename(self.save_dir / "validation")
        (self.save_dir / "minecraft/train").rename(self.save_dir / "training")
        (self.save_dir / "minecraft").rmdir()
        for part_suffix in part_suffixes:
            identifier = f"minecraft_marsh_dataset_{part_suffix}"
            file_name = f"minecraft.tar.part{part_suffix}"
            part_file = self.save_dir / identifier / file_name
            part_file.rmdir()

    def video_length(self, video_metadata: Dict[str, Any]) -> int:
        return 300

    def build_transform(self):
        return transforms.Resize(
            (self.resolution, self.resolution),
            interpolation=InterpolationMode.NEAREST_EXACT,
            antialias=True,
        )


class MinecraftSimpleVideoDataset(MinecraftBaseVideoDataset, BaseSimpleVideoDataset):
    """
    Minecraft simple video dataset
    """

    def __init__(self, cfg: DictConfig, split: str = "training"):
        if split == "test":
            split = "validation"
        BaseSimpleVideoDataset.__init__(self, cfg, split)


class MinecraftAdvancedVideoDataset(
    MinecraftBaseVideoDataset, BaseAdvancedVideoDataset
):
    """
    Minecraft advanced video dataset
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: str = "training",
        current_epoch: Optional[int] = None,
    ):
        if split == "test":
            split = "validation"
        BaseAdvancedVideoDataset.__init__(self, cfg, split, current_epoch)

    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        path = video_metadata["video_paths"].with_suffix(".npz")
        actions = np.load(path)["actions"][start_frame:end_frame]
        return torch.from_numpy(np.eye(4)[actions]).float()
