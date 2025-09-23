from datasets.video import (
    MinecraftSimpleVideoDataset,
)
from algorithms.vae import ImageVAEPreprocessor
from .base_exp import BaseLightningExperiment
from .data_modules import ValDataModule


class VideoLatentPreprocessingExperiment(BaseLightningExperiment):
    """
    Experiment for preprocessing videos to latents using a pretrained ImageVAE model.
    """

    compatible_algorithms = dict(
        image_vae_preprocessor=ImageVAEPreprocessor,
    )

    compatible_datasets = dict(
        video_minecraft=MinecraftSimpleVideoDataset,
    )

    data_module_cls = ValDataModule

    def training(self) -> None:
        raise NotImplementedError(
            "Training not implemented for video preprocessing experiments"
        )

    def testing(self) -> None:
        raise NotImplementedError(
            "Testing not implemented for video preprocessing experiments"
        )
