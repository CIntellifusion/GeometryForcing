import torch
from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan
from .base_data_module import BaseDataModule


class ResumableDataModule(BaseDataModule):
    """
    A resumable data module for `datasets.video.base_iterable_video_dataset.BaseAdvancedVideoDataset`.
    Activated when `experiment.reload_dataloaders_every_n_epochs = 1`, `experiment.training.data.shuffle = False`, checkpointing & validation are epoch-based, and `dataset.subdataset_size` is set.
    Data module will pass the current epoch to the dataset, so that the dataset can deterministically compute the subdataset corresponding to the current epoch.
    """

    @property
    def is_resumable(self) -> bool:
        is_experiment_resumable = (
            self.root_cfg.experiment.reload_dataloaders_every_n_epochs == 1
            and not self.exp_cfg.training.data.shuffle
            and self.root_cfg.experiment.training.checkpointing.every_n_epochs
            is not None
            and self.root_cfg.experiment.validation.val_every_n_epoch is not None
            and self.root_cfg.experiment.training.max_steps == -1
            and (
                self.root_cfg.experiment.validation.val_every_n_epoch > 1
                or self.root_cfg.experiment.validation.val_every_n_step == 1.0
            )  # this ensures that ckpts are saved not before validation (otherwise,it may lead to redundant ckpts & validation if resuming)
        )
        is_dataset_resumable = self.root_cfg.dataset.subdataset_size is not None
        if is_experiment_resumable != is_dataset_resumable:
            raise ValueError(
                "To make a resumable run, set experiment.reload_dataloaders_every_n_epochs = 1, experiment.training.data.shuffle = False, checkpointing & validation are epoch-based, and dataset.subdataset_size is set."
            )
        return is_experiment_resumable and is_dataset_resumable

    def _build_dataset(self, split: str) -> torch.utils.data.Dataset:
        if split in ["training", "test", "validation"]:
            is_resumable = self.is_resumable and split == "training"
            dataset = self.compatible_datasets[self.root_cfg.dataset._name](
                self.root_cfg.dataset,
                split=split,
                current_epoch=(self.trainer.current_epoch if is_resumable else None),
            )

            if is_resumable:
                rank_zero_print(
                    cyan(
                        f"Resumable Training ({(dataset.cumulative_sizes[-1] / dataset.subdataset_size):.1f} subepochs / epoch)"
                    ),
                    f"currently at subepoch {dataset.current_subepoch}",
                )

            return dataset
        else:
            raise NotImplementedError(f"split '{split}' is not implemented")
