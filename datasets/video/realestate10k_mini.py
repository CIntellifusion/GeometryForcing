from typing import Optional
from omegaconf import DictConfig
from utils.print_utils import cyan
from torchvision.datasets.utils import (
    download_and_extract_archive,
)
from .base_video import SPLIT
from .realestate10k import RealEstate10KAdvancedVideoDataset


class RealEstate10KMiniAdvancedVideoDataset(RealEstate10KAdvancedVideoDataset):
    """
    A mini version of the RealEstate10K video dataset,
    containing 500 test set videos (>= 200 frames each).
    """

    _DATASET_URL = "https://huggingface.co/kiwhansong/DFoT/resolve/main/datasets/RealEstate10K_Mini.tar.gz"

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ):
        assert (
            split != "training"
        ), "RealEstate10KMiniAdvancedVideoDataset is only for evaluation"
        super().__init__(cfg, split, current_epoch)

    def _should_download(self) -> bool:
        return not self.save_dir.exists()

    def download_dataset(self):
        print(cyan("Downloading RealEstate10k Mini dataset..."))
        download_and_extract_archive(
            self._DATASET_URL,
            self.save_dir.parent,
            remove_finished=True,
        )
        print(cyan("Finished downloading RealEstate10k Mini dataset!"))
