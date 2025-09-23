from typing import Literal, List, Dict, Any, Callable, Tuple, Optional
from abc import ABC, abstractmethod
import random
import bisect
from pathlib import Path
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torchvision.datasets.video_utils import _VideoTimestampsDataset, _collate_fn
from tqdm import tqdm
from einops import rearrange
from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan
from datasets.video.utils import read_video, VideoTransform
import os 
SPLIT = Literal["training", "validation", "test"]


class BaseVideoDataset(torch.utils.data.Dataset, ABC):
    """
    Common base class for video dataset.
    Methods here are shared between simple and advanced video datasets

    Folder structure of each dataset:
    - {save_dir} (specified in config, e.g., data/phys101)
        - /{split}
            - data files (e.g. 000001.mp4, 000001.pt)
        - /metadata
            - {split}.pt
    - {save_dir}_latent_{latent_resolution} (same structure as save_dir)
    """

    _ALL_SPLITS = ["training", "validation", "test"]
    metadata: Dict[str, Any]

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
    ):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.resolution = cfg.resolution
        self.latent_resolution = cfg.resolution // cfg.latent.downsampling_factor[1]
        self.save_dir = Path(cfg.save_dir)
        self.latent_dir = self.save_dir.with_name(
            f"{self.save_dir.name}_latent_{self.latent_resolution}{'_' + cfg.latent.suffix if cfg.latent.suffix else ''}"
        )
        self.split_dir = self.save_dir / split
        self.metadata_dir = self.save_dir / "metadata"

        # Download dataset if not exists
        if self._should_download():
            self.download_dataset()
        if not self.metadata_dir.exists():
            self.metadata_dir.mkdir(exist_ok=True, parents=True)
            for split in self._ALL_SPLITS:
                self.build_metadata(split)

        self.metadata = self.load_metadata()
        print(f"Loaded {len(self.metadata)} videos for {self.split} split")
        self.augment_dataset()
        self.transform = self.build_transform()

    def _should_download(self) -> bool:
        """
        Check if the dataset should be downloaded
        """
        return not (self.save_dir / self.split).exists()

    @abstractmethod
    def download_dataset(self) -> None:
        """
        Download dataset from the internet and build it in save_dir
        """
        raise NotImplementedError

    def build_metadata(self, split: SPLIT) -> None:
        """
        Build metadata for the dataset and save it in metadata_dir
        This may vary depending on the dataset.
        Default:
        ```
        {
            "video_paths": List[str],
            "video_pts": List[str],
            "video_fps": List[float],
        }
        ```
        """
        video_paths = sorted(list((self.save_dir / split).glob("**/*.mp4")), key=str)
        dl: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            _VideoTimestampsDataset(video_paths),
            batch_size=16,
            num_workers=64,
            collate_fn=_collate_fn,
        )
        video_pts: List[torch.Tensor] = (
            []
        )  # each entry is a tensor of shape (num_frames, )
        video_fps: List[float] = []

        with tqdm(total=len(dl), desc=f"Building metadata for {split}") as pbar:
            for batch in dl:
                pbar.update(1)
                batch_pts, batch_fps = list(zip(*batch))
                batch_pts = [
                    torch.as_tensor(pts, dtype=torch.long) for pts in batch_pts
                ]
                video_pts.extend(batch_pts)
                video_fps.extend(batch_fps)

        metadata = {
            "video_paths": video_paths,
            "video_pts": video_pts,
            "video_fps": video_fps,
        }
        torch.save(metadata, self.metadata_dir / f"{split}.pt")

    def subsample(
        self,
        metadata: List[Dict[str, Any]],
        filter_fn: Callable[[Dict[str, Any]], bool],
        filter_msg: str,
    ) -> List[Dict[str, Any]]:
        """
        Subsample the dataset with the given filter function
        """
        before_len = len(metadata)
        metadata = [
            video_metadata for video_metadata in metadata if filter_fn(video_metadata)
        ]
        after_len = len(metadata)
        rank_zero_print(
            cyan(
                f"{self.split}: {after_len} / {before_len} videos will be used after filtering out {filter_msg}"
            ),
        )
        return metadata

    def augment_dataset(self) -> None:
        """
        Augment the dataset
        """
        # pylint: disable=assignment-from-none
        augmentation = self._build_data_augmentation()
        if augmentation is not None:
            self.metadata = self._augment_dataset(self.metadata, *augmentation)

    def _augment_dataset(
        self,
        metadata: List[Dict[str, Any]],
        augment_fn: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
        augment_msg: str,
    ) -> List[Dict[str, Any]]:
        """
        Augment the dataset (corresponds to metadata) with the given augment function

        Args:
            metadata: list of video metadata - stands for the dataset
            augment_fn: function that takes video metadata and returns a list of augmented video metadata
        """
        before_len = len(metadata)
        metadata = [
            augmented_video_metadata
            for video_metadata in metadata
            for augmented_video_metadata in augment_fn(video_metadata)
        ]
        after_len = len(metadata)
        rank_zero_print(
            cyan(
                f"{self.split}: {before_len} -> {after_len} videos after augmenting with {augment_msg}"
            ),
        )
        return metadata

    def _build_data_augmentation(
        self,
    ) -> Optional[Tuple[Callable[[Dict[str, Any]], List[Dict[str, Any]]], str]]:
        """
        Build a data augmentation (composed of augment_fn and augment_msg) that will be applied to the dataset

        if None, no data augmentation will be applied
        """
        return None

    def load_metadata(self) -> List[Dict[str, Any]]:
        """
        Load metadata from metadata_dir
        """
        metadata = torch.load(
            self.metadata_dir / f"{self.split}.pt", weights_only=False
        )
        return [
            {key: metadata[key][i] for key in metadata.keys()}
            for i in range(len(metadata["video_paths"])) if os.path.exists(metadata["video_paths"][i]) 
        ]

    def video_length(self, video_metadata: Dict[str, Any]) -> int:
        """
        Return the length of the video at idx
        """
        return len(video_metadata["video_pts"])

    def build_transform(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Build a transform that will be applied to each video frame.
        (e.g. resize, center crop)
        """
        return VideoTransform((self.resolution, self.resolution))

    def video_metadata_to_latent_path(self, video_metadata: Dict[str, Any]) -> Path:
        """
        Convert video_path to latent_path
        """
        return (
            self.latent_dir / video_metadata["video_paths"].relative_to(self.save_dir)
        ).with_suffix(".pt")

    def get_latent_paths(self, split: SPLIT) -> List[Path]:
        """
        Return list of latent paths for the given split
        """
        return sorted(list((self.latent_dir / split).glob("**/*.pt")), key=str)

    def load_video(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Load video from video_idx with given start_frame and end_frame (exclusive)
        if end_frame is None, load until the end of the video
        return shape: (T, C, H, W)
        """
        if end_frame is None:
            end_frame = self.video_length(video_metadata)
        video_path, video_pts = (
            video_metadata["video_paths"],
            video_metadata["video_pts"],
        )
        start_pts = video_pts[start_frame].item()
        end_pts = video_pts[end_frame - 1].item()
        video = read_video(video_path, start_pts, end_pts)
        return video.permute(0, 3, 1, 2) / 255.0


class BaseSimpleVideoDataset(BaseVideoDataset):
    """
    Base class for simple video datasets
    that load full videos with given resolution
    Also provides latent_path where latent should be saved
    """

    def __init__(self, cfg: DictConfig, split: SPLIT = "training"):
        super().__init__(cfg, split)
        self.latent_dir.mkdir(exist_ok=True, parents=True)
        # filter videos to only include the ones that have not been preprocessed
        self.metadata = self.exclude_videos_with_latents(self.metadata)

    def exclude_videos_with_latents(
        self, metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        latent_paths = set(self.get_latent_paths(self.split))

        return self.subsample(
            metadata,
            lambda video_metadata: self.video_metadata_to_latent_path(video_metadata)
            not in latent_paths,
            "videos that have already been preprocessed to latents",
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        loads video together with the path where latent should be saved
        """
        video_metadata = self.metadata[idx]
        video = self.load_video(video_metadata, 0)

        return (
            self.transform(video),
            self.video_metadata_to_latent_path(video_metadata).as_posix(),
        )


class BaseAdvancedVideoDataset(BaseVideoDataset):
    """
    Base class for video dataset
    that load video clips with given resolution and frame skip
    Videos may be of variable lengths.
    """

    cumulative_sizes: List[int]
    idx_remap: List[int]

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ):
        super().__init__(cfg, split)
        self.use_preprocessed_latents = (
            cfg.latent.enable and cfg.latent.type.startswith("pre_")
        )
        self.current_subepoch = current_epoch
        self.subdataset_size = cfg.subdataset_size

        if self.use_preprocessed_latents and not self.latent_dir.exists():
            raise ValueError(
                f"Preprocess the video to latents first and save them in {self.latent_dir}"
            )

        self.external_cond_dim = cfg.external_cond_dim * (
            cfg.frame_skip if cfg.external_cond_stack else 1
        )
        self.n_frames = (
            1
            + ((cfg.max_frames if split == "training" else cfg.n_frames) - 1)
            * cfg.frame_skip
        )
        self.frame_skip = cfg.frame_skip

        if self.use_preprocessed_latents:
            self.metadata = self.exclude_videos_without_latents(self.metadata)

        if split == "training" or cfg.filter_min_len is None:
            self.filter_min_len = self.n_frames
        else:
            self.filter_min_len = cfg.filter_min_len

        self.metadata = self.exclude_short_videos(self.metadata, self.filter_min_len)

        self.on_before_prepare_clips()
        # When overriding __init__ in a subclass,
        # no more dataset filtering should be performed after prepare_clips is called (i.e. after super().__init__())
        # Instead, override on_before_prepare_clips to perform additional modifications before prepare_clips
        self.prepare_clips()

    @property
    def use_subdataset(self) -> bool:
        """
        Check if subdataset strategy is enabled
        """
        return (
            self.split == "training"
            and self.subdataset_size is not None
            and self.current_subepoch is not None
        )

    @property
    def use_evaluation_subdataset(self) -> bool:
        """
        Check if using deterministic subdataset for evaluation
        """
        return self.split != "training" and self.cfg.num_eval_videos is not None

    def on_before_prepare_clips(self) -> None:
        """
        Additional setup before preparing clips (e.g. excluding invalid videos)
        """
        return

    def prepare_clips(self) -> None:
        """
        Compute cumulative sizes for the dataset and update self.cumulative_sizes
        Shuffle the dataset with a fixed seed
        """
        num_clips = torch.as_tensor(
            [
                max(self.video_length(video_metadata) - self.n_frames + 1, 1)
                for video_metadata in self.metadata
            ]
        )
        self.cumulative_sizes = num_clips.cumsum(0).tolist()
        self.idx_remap = self._build_idx_remap()

    def _build_idx_remap(self) -> List[int]:
        """
        Deterministically build idx_remap for the dataset, which maps the indices of the current dataset to the absolute indices of the full dataset
        - If use_subdataset is True, idx_remap remaps the subdataset indices (ranging from 0 to self.subdataset_size) to the full dataset indices (ranging from 0 to self.cumulative_sizes[-1])
        - If use_evaluation_subdataset is True, idx_remap remaps the indices (range from 0 to self.cfg.num_eval_videos) to the full dataset indices (ranging from 0 to self.cumulative_sizes[-1]), where each index corresponds to a randomly chosen clip from randomly chosen num_eval_videos videos
        - Otherwise, idx_remap is a deterministic shuffle of 0 to self.__len__()
        """

        if self.use_subdataset:
            # assign deterministic sequence of indices to each subepoch
            def idx_to_epoch_and_idx(idx: int) -> Tuple[int, int]:
                effective_idx = idx + self.subdataset_size * self.current_subepoch
                return divmod(effective_idx, self.cumulative_sizes[-1])

            start_epoch, start_idx_in_epoch = idx_to_epoch_and_idx(0)
            end_epoch, end_idx_in_epoch = idx_to_epoch_and_idx(self.subdataset_size - 1)
            assert (
                0 <= end_epoch - start_epoch <= 1
            ), f"Subdataset size should be <= dataset size got end_epoch={end_epoch}, start_epoch={start_epoch}"

            epoch_to_shuffled_indices: Dict[int, List[int]] = {}
            for epoch in range(start_epoch, end_epoch + 1):
                indices = list(range(self.cumulative_sizes[-1]))
                random.seed(epoch)
                random.shuffle(indices)
                epoch_to_shuffled_indices[epoch] = indices

            if start_epoch == end_epoch:
                idx_remap = epoch_to_shuffled_indices[start_epoch][
                    start_idx_in_epoch : end_idx_in_epoch + 1
                ]
            else:
                idx_remap = (
                    epoch_to_shuffled_indices[start_epoch][start_idx_in_epoch:]
                    + epoch_to_shuffled_indices[end_epoch][: end_idx_in_epoch + 1]
                )
            assert (
                len(idx_remap) == self.subdataset_size
            ), "Something went wrong while remapping subdataset indices"
            return idx_remap
        elif self.use_evaluation_subdataset:
            # deterministically choose one clip per video for evaluation
            # raise a warning if num_eval_videos > num_videos
            if self.cfg.num_eval_videos > len(self.cumulative_sizes):
                rank_zero_print(
                    cyan(
                        f"There are less clips ({len(self.cumulative_sizes)}) in the dataset than the number of requested evaluation clips ({self.cfg.num_eval_videos})"
                    )
                )
            random.seed(0)
            idx_remap = []
            for start, end in zip(
                [0] + self.cumulative_sizes[:-1], self.cumulative_sizes
            ):
                idx_remap.append(random.randrange(start, end))
            random.shuffle(idx_remap)
            return idx_remap[: self.cfg.num_eval_videos]

        else:
            # shuffle but keep the same order for each epoch, so validation sample is diverse yet deterministic
            idx_remap = list(range(self.__len__()))
            random.seed(0)
            random.shuffle(idx_remap)
            return idx_remap

    def exclude_short_videos(
        self, metadata: List[Dict[str, Any]], min_frames: int
    ) -> List[Dict[str, Any]]:
        """
        Exclude videos that are shorter than n_frames
        """

        return self.subsample(
            metadata,
            lambda video_metadata: self.video_length(video_metadata) >= min_frames,
            f"videos shorter than {min_frames} frames",
        )

    def exclude_videos_without_latents(
        self, metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        latent_paths = set(self.get_latent_paths(self.split))
        return self.subsample(
            metadata,
            lambda video_metadata: self.video_metadata_to_latent_path(video_metadata)
            in latent_paths,
            "videos without latents",
        )

    def get_clip_location(self, idx: int) -> Tuple[int, int]:
        idx = self.idx_remap[idx]
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]
        return video_idx, clip_idx

    def load_latent(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        latent_path = self.video_metadata_to_latent_path(video_metadata)
        return torch.load(latent_path, weights_only=False)[start_frame:end_frame]

    @abstractmethod
    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        raise NotImplementedError

    def load_video_and_cond(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load video and conditions from video_idx with given start_frame and end_frame (exclusive)
        """
        video = self.load_video(video_metadata, start_frame, end_frame)
        cond = self.load_cond(video_metadata, start_frame, end_frame)
        return video, cond

    def __len__(self) -> int:
        return (
            self.subdataset_size
            if self.use_subdataset
            else (
                min(self.cfg.num_eval_videos, len(self.cumulative_sizes))
                if self.use_evaluation_subdataset
                else self.cumulative_sizes[-1]
            )
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_idx, clip_idx = self.get_clip_location(idx)
        video_metadata = self.metadata[video_idx]
        video_length = self.video_length(video_metadata)
        start_frame, end_frame = clip_idx, min(clip_idx + self.n_frames, video_length)

        video, latent, cond = None, None, None
        if self.use_preprocessed_latents:
            latent = self.load_latent(video_metadata, start_frame, end_frame)

        if self.use_preprocessed_latents and self.split == "training":
            # do not load video if we are training with latents
            if self.external_cond_dim > 0:
                cond = self.load_cond(video_metadata, start_frame, end_frame)

        else:
            if self.external_cond_dim > 0:
                # load video together with condition
                video, cond = self.load_video_and_cond(
                    video_metadata, start_frame, end_frame
                )
            else:
                # load video only
                video = self.load_video(video_metadata, start_frame, end_frame)

        lens = [len(x) for x in (video, cond, latent) if x is not None]
        assert len(set(lens)) == 1, f"video, cond, latent must have the same length {lens}"
        pad_len = self.n_frames - lens[0]

        nonterminal = torch.ones(self.n_frames, dtype=torch.bool)
        if pad_len > 0:
            if video is not None:
                video = F.pad(video, (0, 0, 0, 0, 0, 0, 0, pad_len)).contiguous()
            if latent is not None:
                latent = F.pad(latent, (0, 0, 0, 0, 0, 0, 0, pad_len)).contiguous()
            if cond is not None:
                cond = F.pad(cond, (0, 0, 0, pad_len)).contiguous()
            nonterminal[-pad_len:] = 0

        if self.frame_skip > 1:
            if video is not None:
                video = video[:: self.frame_skip]
            if latent is not None:
                latent = latent[:: self.frame_skip]
            nonterminal = nonterminal[:: self.frame_skip]
        if cond is not None:
            cond = self._process_external_cond(cond)

        output = {
            "videos": self.transform(video) if video is not None else None,
            "latents": latent,
            "conds": cond, # 256 16
            "nonterminal": nonterminal,
        }
        return {key: value for key, value in output.items() if value is not None}

    def _process_external_cond(self, external_cond: torch.Tensor) -> torch.Tensor:
        """
        Post-processes external condition.
        Args:
            external_cond: (T, *) tensor, T = self.n_frames
        Returns:
            processed_cond: (T', *) tensor, T' = number of frames after frame skip
        By default:
            shifts external condition by self.frame_skip - 1
            so that each frame has condition corresponding to
            current and previous frames
            then stacks the conditions for skipping frames
        """
        if self.frame_skip == 1:
            return external_cond
        external_cond = F.pad(external_cond, (0, 0, self.frame_skip - 1, 0), value=0.0)
        return rearrange(external_cond, "(t fs) d -> t (fs d)", fs=self.frame_skip)
