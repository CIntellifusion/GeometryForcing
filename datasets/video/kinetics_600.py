"""
Adapted from https://github.com/pytorch/vision/blob/main/torchvision/datasets/kinetics.py
"""

from typing import Any, Dict, List, Optional, Literal
from fractions import Fraction
import csv
import os
import random
from os import path
from pathlib import Path
import urllib
import shutil
from multiprocessing import Pool
from functools import partial
from omegaconf import DictConfig
import torch
from torchvision.io import write_video
from torchvision.datasets.utils import (
    download_and_extract_archive,
    check_integrity,
    download_url,
)
from tqdm import tqdm
import numpy as np
from utils.print_utils import cyan
from .base_video import (
    BaseVideoDataset,
    BaseSimpleVideoDataset,
    BaseAdvancedVideoDataset,
    SPLIT,
)
from .utils import read_video, rescale_and_crop

VideoPreprocessingType = Literal["npz", "mp4"]
VideoPreprocessingMp4FPS: int = 10


def _dl_wrap(tarpath: str, videopath: str, line: str) -> None:
    download_and_extract_archive(line, tarpath, videopath, remove_finished=True)


def _preprocess_video(
    video_path: Path,
    resolution: int,
    preprocessing_type: VideoPreprocessingType = "npz",
):
    try:
        video = read_video(str(video_path))
        video = rescale_and_crop(video, resolution)
        video_path = (
            video_path.parent.parent
            / f"{video_path.parent.name}_preprocessed_{resolution}_{preprocessing_type}"
            / video_path.name
        ).with_suffix("." + preprocessing_type)

        if preprocessing_type == "npz":
            np.savez_compressed(
                video_path,
                video=video.transpose(0, 3, 1, 2).copy(),
            )
        elif preprocessing_type == "mp4":
            write_video(
                filename=video_path,
                video_array=torch.from_numpy(video).clone(),
                fps=VideoPreprocessingMp4FPS,
            )

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
    # remove original video
    # video_path.unlink()


class Kinetics600BaseVideoDataset(BaseVideoDataset):
    _TAR_URL = "https://s3.amazonaws.com/kinetics/600/{split}/k600_{split}_path.txt"
    _ANNOTATION_URLS = "https://s3.amazonaws.com/kinetics/600/annotations/{split}.csv"

    @property
    def use_video_preprocessing(self) -> bool:
        return self.cfg.video_preprocessing is not None

    def _should_download(self) -> bool:
        return super()._should_download() and not any(
            (
                self.save_dir
                / f"{self.split}_preprocessed_{self.resolution}_{video_preprocessing}"
            ).exists()
            for video_preprocessing in ["npz", "mp4"]
        )

    def download_dataset(self) -> None:
        print(cyan("Downloading Kinetics600 dataset..."))
        for split in ["train", "val", "test"]:
            self._download_videos(split)
            # self._make_ds_structure(split) # disabled since we are training unconditional models

        (self.save_dir / "train").rename(self.save_dir / "training")
        (self.save_dir / "val").rename(self.save_dir / "validation")
        shutil.rmtree(self.save_dir / "tars")
        shutil.rmtree(self.save_dir / "files")
        print(cyan("Finished downloading Kinetics600 dataset!"))

    def _download_videos(self, split: SPLIT) -> None:
        print(cyan(f"Downloading {split} videos..."))
        split_folder = self.save_dir / split
        tar_path = self.save_dir / "tars"
        file_list_path = self.save_dir / "files"

        split_url = self._TAR_URL.format(split=split)
        split_url_filepath = file_list_path / path.basename(split_url)
        if not check_integrity(split_url_filepath):
            download_url(split_url, file_list_path)
        with open(split_url_filepath) as file:
            list_video_urls = [
                urllib.parse.quote(line, safe="/,:")
                for line in file.read().splitlines()
            ]

        part = partial(_dl_wrap, tar_path, split_folder)
        with Pool(32) as pool:
            list(
                tqdm(
                    pool.imap(part, list_video_urls),
                    total=len(list_video_urls),
                    desc=f"Downloading {split} videos",
                )
            )

    def _make_ds_structure(self, split) -> None:
        """move videos from
        split_folder/
            ├── clip1.avi
            ├── clip2.avi

        to the correct format as described below:
        split_folder/
            ├── class1
            │   ├── clip1.avi

        """
        print(cyan(f"Organizing {split} videos..."))
        split_folder = path.join(self.save_dir, split)
        annotation_path = path.join(self.save_dir, "annotations")
        if not check_integrity(path.join(annotation_path, f"{split}.csv")):
            download_url(
                self._ANNOTATION_URLS.format(split=split),
                annotation_path,
            )
        annotations = path.join(annotation_path, f"{split}.csv")

        file_fmtstr = "{ytid}_{start:06}_{end:06}.mp4"
        with open(annotations) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                f = file_fmtstr.format(
                    ytid=row["youtube_id"],
                    start=int(row["time_start"]),
                    end=int(row["time_end"]),
                )
                label = (
                    row["label"]
                    .replace(" ", "_")
                    .replace("'", "")
                    .replace("(", "")
                    .replace(")", "")
                )
                os.makedirs(path.join(split_folder, label), exist_ok=True)
                downloaded_file = path.join(split_folder, f)
                if path.isfile(downloaded_file):
                    os.replace(
                        downloaded_file,
                        path.join(split_folder, label, f),
                    )

    def setup(self) -> None:
        if self.use_video_preprocessing:
            if not (
                self.save_dir
                / f"{self.split}_preprocessed_{self.resolution}_{self.cfg.video_preprocessing}"
            ).exists():
                for split in ["training", "validation", "test"]:
                    self._preprocess_videos(split)
            self.metadata = self.exclude_failed_videos(self.metadata)
            self.transform = lambda x: x

    def _preprocess_videos(self, split: SPLIT) -> None:
        """
        Preprocesses videos to {self.resolution}x{self.resolution} resolution
        """
        print(
            cyan(
                f"Preprocessing {split} videos to {self.resolution}x{self.resolution}..."
            )
        )
        (
            self.save_dir
            / f"{split}_preprocessed_{self.resolution}_{self.cfg.video_preprocessing}"
        ).mkdir(parents=True, exist_ok=True)
        video_paths = torch.load(self.metadata_dir / f"{split}.pt", weights_only=False)
        video_paths = video_paths["video_paths"]
        preprocess_fn = partial(
            _preprocess_video,
            resolution=self.resolution,
            preprocessing_type=self.cfg.video_preprocessing,
        )
        with Pool(32) as pool:
            list(
                tqdm(
                    pool.imap(preprocess_fn, video_paths),
                    total=len(video_paths),
                    desc=f"Preprocessing {split} videos",
                )
            )

    def exclude_failed_videos(
        self, metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Exclude videos that failed to preprocess
        """
        preprocessed_video_paths = set(
            list(
                (
                    self.save_dir
                    / f"{self.split}_preprocessed_{self.resolution}_{self.cfg.video_preprocessing}"
                ).glob(f"**/*.{self.cfg.video_preprocessing}")
            )
        )
        return self.subsample(
            metadata,
            lambda video_metadata: self.video_path_to_preprocessed_path(
                video_metadata["video_paths"]
            )
            in preprocessed_video_paths,
            "failed-to-preprocess videos",
        )

    def video_path_to_preprocessed_path(self, video_path: Path) -> Path:
        return (
            video_path.parent.parent
            / f"{video_path.parent.name}_preprocessed_{self.resolution}_{self.cfg.video_preprocessing}"
            / video_path.name
        ).with_suffix("." + self.cfg.video_preprocessing)

    def load_video(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        if self.use_video_preprocessing:
            preprocessed_path = self.video_path_to_preprocessed_path(
                video_metadata["video_paths"]
            )
            match self.cfg.video_preprocessing:
                case "npz":
                    video = np.load(
                        preprocessed_path,
                    )[
                        "video"
                    ][start_frame:end_frame]
                    return torch.from_numpy(video / 255.0).float()
                case "mp4":
                    video = read_video(
                        preprocessed_path,
                        pts_unit="sec",
                        start_pts=Fraction(start_frame, VideoPreprocessingMp4FPS),
                        end_pts=Fraction(end_frame - 1, VideoPreprocessingMp4FPS),
                    )
                    return video.permute(0, 3, 1, 2) / 255.0
        else:
            return super().load_video(video_metadata, start_frame, end_frame)


class Kinetics600SimpleVideoDataset(
    Kinetics600BaseVideoDataset, BaseSimpleVideoDataset
):
    """
    Kinetics-600 simple video dataset
    """

    def __init__(self, cfg: DictConfig, split: SPLIT = "training"):
        BaseSimpleVideoDataset.__init__(self, cfg, split)
        self.setup()


class Kinetics600AdvancedVideoDataset(
    Kinetics600BaseVideoDataset, BaseAdvancedVideoDataset
):
    """
    Kinetics-600 advanced video dataset
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ):
        if split == "validation":
            split = "test"
        BaseAdvancedVideoDataset.__init__(self, cfg, split, current_epoch)

    def on_before_prepare_clips(self) -> None:
        self.setup()

    def setup(self) -> None:
        super().setup()

    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        raise NotImplementedError("kinetics-600 only supports unconditional models")
