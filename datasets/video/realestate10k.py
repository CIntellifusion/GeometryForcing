from typing import Tuple, List, Dict, Any, Literal, Optional
from fractions import Fraction
from pathlib import Path
import random
import io
from multiprocessing import Pool
import subprocess
from functools import partial
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
import torch
from torchvision.io import write_video
from torchvision.datasets.utils import (
    download_and_extract_archive,
)
import numpy as np
from pytubefix import YouTube
from utils.print_utils import cyan
from utils.storage_utils import safe_torch_save
from .base_video import (
    BaseVideoDataset,
    BaseSimpleVideoDataset,
    BaseAdvancedVideoDataset,
    SPLIT,
)
from .utils import read_video, rescale_and_crop, random_bool

VideoPreprocessingType = Literal["npz", "mp4"]
VideoPreprocessingMp4FPS: int = 10
DownloadPlan = Dict[str, Dict[str, List[float]]]


class RealEstate10KBaseVideoDataset(BaseVideoDataset):
    """
    RealEstate10K base video dataset.
    The dataset will be preprocessed to `_SUPPORTED_RESOLUTIONS` in the format of `_SUPPORTED_RESOLUTIONS[resolution]` during the download.
    """

    _ALL_SPLITS = ["training", "test"]
    # this originally comes from https://storage.cloud.google.com/realestate10k-public-files/RealEstate10K.tar.gz
    # but is served at the above URL to avoid the need for manual download + place in the right directory
    _DATASET_URL = "https://huggingface.co/kiwhansong/DFoT/resolve/main/datasets/RealEstate10K.tar.gz"
    _SUPPORTED_RESOLUTIONS: Dict[int, VideoPreprocessingType] = {
        64: "npz",
        256: "mp4",
    }

    def _should_download(self) -> bool:
        return False 
        if self.resolution not in self._SUPPORTED_RESOLUTIONS:
            raise ValueError(
                f"Resolution {self.resolution} is not supported. Supported resolutions: {list(self._SUPPORTED_RESOLUTIONS.keys())}. Please modify `_SUPPORTED_RESOLUTIONS` in the RealEstate10kBaseVideoDataset class to support this resolution."
            )

        return (
            super()._should_download()
            and not (self.save_dir / f"{self.split}_{self.resolution}").exists()
        )

    def download_dataset(self) -> None:
        print(cyan("Downloading RealEstate10k dataset..."))
        print(
            cyan(
                "Please read the NOTE in the `_download_videos` function at `datasets/video/realestate10k_video_dataset.py` before continuing."
            )
        )
        # print(cyan("Shortcutting this function will not download the dataset!"))
        # return 
        input("Press Enter to continue...")
        # download_and_extract_archive(
        #     self._DATASET_URL,
        #     self.save_dir,
        #     filename="raw.tar.gz",
        #     remove_finished=True,
        # )
        try:
            (self.save_dir / "RealEstate10K").rename(self.save_dir / "raw")
            (self.save_dir / "raw" / "train").rename(self.save_dir / "raw" / "training")
        except OSError:
            print(
                "The dataset is already downloaded. Please remove the existing dataset before downloading again."
            )
        
        for split in ["training", "test"]:
            plan = self._build_download_plan(split)
            # import pdb; pdb.set_trace() 
            self._download_videos(split, plan)
            self._preprocess_videos(split, plan)

        print(cyan("Finished downloading RealEstate10k dataset!"))

    def _download_videos(self, split: SPLIT, urls: List[str]) -> None:
        """
        NOTE: The RealEstate10k dataset is a collection of YouTube videos, and downloading them should be done with caution to ensure that the dataset is not lost.

        This function may fail due to the following reasons:
        - The video is not available anymore, deleted, or private on YouTube. In this case, you shall ignore the error and continue.
        - Bot detection from YouTube. You may meet the error "This request was detected as a bot. Use `use_po_token=True` to view. See more details at https://github.com/JuanBindez/pytubefix/pull/209". When this happens, you will miss all videos afterward, so you should try to resolve this by using `use_po_token=True` or `use_oauth=True`, or by using a proxy, or by giving a time delay between each download.

        The exact size of the RealEstate10k dataset may change across time as it relies on the availability of YouTube videos, so we provide the following statistics as of the time of our own download (You should expect a similar but not identical size):
        - training: 6132 / 6559 videos = 65798 / 71556 clips -> 65725 clips after preprocessing
        - test: 655 / 696 videos = 7148 / 7711 clips
        """
        print(cyan(f"Downloading {split} videos from YouTube..."))
        download_dir = self.save_dir / "raw" / split #  PosixPath('data/real-estate-10k/raw/training') 
        download_dir.mkdir(parents=True, exist_ok=True)
        download_fn = partial(_download_youtube_video, download_dir=download_dir)
        with Pool(1) as pool:
            list(
                tqdm(
                    pool.imap(download_fn, urls),
                    total=len(urls),
                    desc=f"Downloading {split} videos",
                )
            )

    def _preprocess_videos(self, split: SPLIT, plan: DownloadPlan) -> None:
        print(
            cyan(
                f"Preprocessing {split} videos to resolutions {list(self._SUPPORTED_RESOLUTIONS.keys())}..."
            )
        )
        args = []
        for youtube_url, key_to_timestamps in plan.items():
            video_path = (
                self.save_dir / "raw" / split / f"{_youtube_url_to_id(youtube_url)}.mp4"
            )
            if not video_path.exists():
                continue
            for key, timestamps in key_to_timestamps.items():
                args.append((key, video_path, timestamps))
        preprocess_fn = partial(
            _preprocess_video,
            resolutions_to_preprocessing=self._SUPPORTED_RESOLUTIONS,
        )
        with Pool(32) as pool:
            list(
                tqdm(
                    pool.imap(preprocess_fn, args),
                    total=len(args),
                    desc=f"Preprocessing {split} videos",
                )
            )

    def _build_download_plan(self, split: SPLIT) -> DownloadPlan:
        """
        Builds a download plan for the specified split.
        Returns a dictionary with the following structure:
            {
                youtube_url: {
                    key: timestamps,
                }
            }
        """
        print(cyan(f"Building download plan & camera poses for {split}..."))
        plan = {}
        txt_files = list((self.save_dir / "raw" / split).glob("*.txt"))
        for txt_file in tqdm(
            txt_files, desc=f"Building download plan & camera poses for {split}"
        ):
            youtube_url, timestamps, cameras = self._read_txt_file(txt_file)
            if youtube_url not in plan:
                plan[youtube_url] = {
                    txt_file.stem: timestamps,
                }
            else:
                plan[youtube_url][txt_file.stem] = timestamps
            safe_torch_save(
                cameras, self.save_dir / f"{split}_poses" / f"{txt_file.stem}.pt"
            )
        return plan

    @staticmethod
    def _read_txt_file(file_path: Path) -> Tuple[str, List[float], torch.Tensor]:
        """
        Reads a txt file containing a video path, a list of timestamps, and a tensor of camera poses.
        """
        timestamps = []
        cameras = []
        youtube_url = ""
        with open(file_path, "r") as f:
            lines = f.readlines()
            assert len(lines) > 0, f"Empty file {file_path}"
            for idx, line in enumerate(lines):
                if idx == 0:
                    youtube_url = line.strip()
                else:
                    timestamp, *camera = line.split(" ")
                    timestamps.append(_timestamp_to_str(int(timestamp)))
                    cameras.append(np.fromstring(",".join(camera), sep=","))
            cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

        return youtube_url, timestamps, cameras

    def build_metadata(self, split: SPLIT) -> None:
        super().build_metadata(f"{split}_256")
        (self.metadata_dir / f"{split}_256.pt").rename(
            self.metadata_dir / f"{split}.pt"
        )

    def setup(self) -> None:
        self.transform = lambda x: x

    def video_path_to_preprocessed_path(self, video_path: Path) -> Path:
        return (
            self.save_dir / f"{self.split}_{self.resolution}" / video_path.name
        ).with_suffix("." + self._SUPPORTED_RESOLUTIONS[self.resolution])

    def load_video(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> torch.Tensor:
        preprocessed_path = self.video_path_to_preprocessed_path(
            video_metadata["video_paths"]
        )
        match preprocessed_path.suffix:
            case ".npz":
                video = np.load(
                    preprocessed_path,
                )[
                    "video"
                ][start_frame:end_frame]
                return torch.from_numpy(video / 255.0).float()
            case ".mp4":
                video = read_video(
                    preprocessed_path,
                    pts_unit="sec",
                    start_pts=Fraction(start_frame, VideoPreprocessingMp4FPS),
                    end_pts=Fraction(end_frame - 1, VideoPreprocessingMp4FPS),
                )
                return video.permute(0, 3, 1, 2) / 255.0


class RealEstate10KSimpleVideoDataset(
    RealEstate10KBaseVideoDataset, BaseSimpleVideoDataset
):
    """
    RealEstate10K simple video dataset
    """

    def __init__(self, cfg: DictConfig, split: SPLIT = "training"):
        BaseSimpleVideoDataset.__init__(self, cfg, split)
        self.setup()


class RealEstate10KAdvancedVideoDataset(
    RealEstate10KBaseVideoDataset, BaseAdvancedVideoDataset
):
    """
    RealEstate10K advanced video dataset
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ):
        if split == "validation":
            split = "test"
        self.maximize_training_data = cfg.maximize_training_data
        self.augmentation = cfg.augmentation
        BaseAdvancedVideoDataset.__init__(self, cfg, split, current_epoch)

    @property
    def _training_frame_skip(self) -> int:
        if self.augmentation.frame_skip_increase == 0:
            return self.frame_skip
        assert (
            self.current_subepoch is not None
        ), "Subepoch should be given to the RealEstate10KAdvancedVideoDataset, to use frame skip schedule"
        return self.frame_skip + int(
            self.current_subepoch * self.augmentation.frame_skip_increase
        )

    def on_before_prepare_clips(self) -> None:
        self.setup()

    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        path = video_metadata["video_paths"]
        path = self.save_dir / f"{self.split}_poses" / f"{path.stem}.pt"
        cond = torch.load(path, weights_only=False)[start_frame:end_frame]
        return cond

    def _augment(
        self, video: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) Horizontal flip augmentation
        if random_bool(self.augmentation.horizontal_flip_prob):
            video = video.flip(-1)
            # NOTE: extrinsics should also be flipped accordingly - the following is equivalent to:
            # E' = I' @ E @ I' where I' = diag([-1, 1, 1, 1]) (E is 4x4 extrinsics matrix)
            cond[:, [5, 6, 7, 8, 12]] *= -1

        # 2) Back-and-forth video augmentation
        # 0 1 2 ... 2k+1 -> 0 2 4 ... 2k 2k+1 ... 3 1
        if random_bool(self.augmentation.back_and_forth_prob):
            video, cond = map(
                lambda x: torch.cat([x[::2], x[1::2].flip(0)], dim=0).contiguous(),
                (video, cond),
            )
        # 3) Reverse video augmentation
        # 0 ... n -> n ... 0
        if random_bool(self.augmentation.reverse_prob):
            video, cond = map(lambda x: x.flip(0).contiguous(), (video, cond))

        return video, cond

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.split != "training":
            return super().__getitem__(idx)

        video_idx, start_frame = self.get_clip_location(idx)
        video_metadata = self.metadata[video_idx]
        video_length = self.video_length(video_metadata)
        frame_skip = (video_length - start_frame - 1) // (self.cfg.max_frames - 1)
        if self.split == "training":
            frame_skip = min(frame_skip, self._training_frame_skip)
        else:
            frame_skip = random.randint(self.frame_skip, frame_skip)

        assert frame_skip > 0, f"Frame skip {frame_skip} should be greater than 0"
        # For training epoch 
        end_frame = start_frame + (self.cfg.max_frames - 1) * frame_skip + 1
        video, cond = self.load_video_and_cond(video_metadata, start_frame, end_frame)
        assert len(video) == len(cond), f"idx={idx} metadata={video_metadata}  Video and cond have different lengths {len(video)} != {len(cond)} video_idx={video_idx} start_frame={start_frame} end_frame={end_frame} frame_skip={frame_skip}"
        video, cond = video[::frame_skip], self._process_external_cond(cond, frame_skip)
        video, cond = self._augment(video, cond)
        return {
            "videos": self.transform(video),
            "conds": cond,
            "nonterminal": torch.ones(self.cfg.max_frames, dtype=torch.bool),
        }

    def exclude_short_videos(
        self, metadata: List[Dict[str, Any]], min_frames: int
    ) -> List[Dict[str, Any]]:
        # if self.maximize_training_data is True,
        # include all videos with at least self.cfg.max_frames frames
        if self.maximize_training_data and self.split == "training":
            min_frames = min(min_frames, self.cfg.max_frames)
        return super().exclude_short_videos(metadata, min_frames)

    def _process_external_cond(
        self, external_cond: torch.Tensor, frame_skip: Optional[int] = None
    ) -> torch.Tensor:
        """
        Converts the raw camera poses to concat-flattened intrinsics and extrinsics.
        Args:
            external_cond (torch.Tensor): Raw camera poses. Shape (T, 18).
            frame_skip (Optional[int]): Frame skip. If None, uses self.frame_skip.
        Returns:
            torch.Tensor: Processed camera poses. Shape (T, 16).
        """
        poses = external_cond[:: frame_skip or self.frame_skip]
        return torch.cat(
            [
                poses[:, :4],
                poses[:, 6:],
            ],
            dim=-1,
        ).to(torch.float32)


def _timestamp_to_str(timestamp: int) -> str:
    timestamp = int(timestamp / 1000)
    str_hour = str(int(timestamp / 3600000)).zfill(2)
    str_min = str(int(int(timestamp % 3600000) / 60000)).zfill(2)
    str_sec = str(int(int(int(timestamp % 3600000) % 60000) / 1000)).zfill(2)
    str_mill = str(int(int(int(timestamp % 3600000) % 60000) % 1000)).zfill(3)
    str_timestamp = str_hour + ":" + str_min + ":" + str_sec + "." + str_mill
    return str_timestamp


def _youtube_url_to_id(youtube_url: str) -> str:
    return youtube_url.split("=")[-1]


def _download_youtube_video(youtube_url: str, download_dir: Path) -> None:
    """
    Downloads a YouTube video to the specified directory.
    Retries with different clients if the download fails, to guarantee that it does not miss any available video.
    """

    def download_with_client(client: Optional[str] = None):
        yt = YouTube(youtube_url,use_po_token=True) if client is None else YouTube(youtube_url, client)
        yt.streams.filter(res="360p").first().download(
            download_dir, filename=f"{_youtube_url_to_id(youtube_url)}.mp4"
        )
    # import pdb; pdb.set_trace()
    try:
        download_with_client(youtube_url)
    except Exception:
        try:
            download_with_client("WEB_EMBED")
        except Exception:
            try:
                download_with_client("IOS")
            except Exception as e:
                print(f"Error downloading {youtube_url}: {e}")


def _read_frame(video_path: Path, timestamp: str) -> torch.Tensor:
    command = [
        "ffmpeg",
        "-ss",
        timestamp,
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "bmp",
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    image_bytes, _ = process.communicate()
    image_stream = io.BytesIO(image_bytes)
    image = Image.open(image_stream)
    image = image.convert("RGB")
    image_array = np.array(image)
    return torch.from_numpy(image_array)


def _preprocess_video(
    info: Tuple[str, Path, List[float]],
    resolutions_to_preprocessing: Dict[int, VideoPreprocessingType],
):
    key, video_path, timestamps = info
    try:
        frames = []
        for timestamp in timestamps:
            frames.append(_read_frame(video_path, timestamp))
        video = torch.stack(frames, dim=0)
        assert video.shape[0] == len(
            timestamps
        ), f"Number of frames {video.shape[0]} does not match the number of timestamps {len(timestamps)} for {key}"

        for resolution, preprocessing_type in resolutions_to_preprocessing.items():
            video_preprocessed = rescale_and_crop(video, resolution)
            save_path = (
                video_path.parent.parent.parent
                / f"{video_path.parent.name}_{resolution}"
                / f"{key}.{preprocessing_type}"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if preprocessing_type == "npz":
                np.savez_compressed(
                    save_path, video=video_preprocessed.transpose(0, 3, 1, 2).copy()
                )
            elif preprocessing_type == "mp4":
                write_video(
                    filename=save_path,
                    video_array=torch.from_numpy(video_preprocessed).clone(),
                    fps=VideoPreprocessingMp4FPS,
                )

    except Exception as e:
        print(f"Error processing {key}: {e}")
