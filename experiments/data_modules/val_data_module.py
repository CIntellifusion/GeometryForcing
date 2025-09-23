from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from .base_data_module import BaseDataModule


class ValDataModule(BaseDataModule):
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return [
            self._dataloader(split) for split in self.exp_cfg.validation.dataset_splits
        ]
