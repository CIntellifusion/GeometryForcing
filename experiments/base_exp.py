"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from abc import ABC
from typing import Optional, Union, Dict
import pathlib

import hydra
import torch
from lightning.pytorch.strategies.ddp import DDPStrategy

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from omegaconf import DictConfig

from utils.print_utils import cyan
from utils.distributed_utils import rank_zero_print
from utils.lightning_utils import EMA
from .data_modules import BaseDataModule

torch.set_float32_matmul_precision("high")


class BaseExperiment(ABC):
    """
    Abstract class for an experiment. This generalizes the pytorch lightning Trainer & lightning Module to more
    flexible experiments that doesn't fit in the typical ml loop, e.g. multi-stage reinforcement learning benchmarks.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms: Dict = NotImplementedError

    def __init__(
        self,
        root_cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        """
        Constructor

        Args:
            cfg: configuration file that contains everything about the experiment
            logger: a pytorch-lightning WandbLogger instance
            ckpt_path: an optional path to saved checkpoint
        """
        super().__init__()
        self.root_cfg = root_cfg
        self.cfg = root_cfg.experiment
        self.debug = root_cfg.debug
        self.logger = logger if logger else False
        self.ckpt_path = ckpt_path
        self.algo = None

    def _build_algo(self):
        """
        Build the lightning module
        :return:  a pytorch-lightning module to be launched
        """
        algo_name = self.root_cfg.algorithm._name
        if algo_name not in self.compatible_algorithms:
            raise ValueError(
                f"Algorithm {algo_name} not found in compatible_algorithms for this Experiment class. "
                "Make sure you define compatible_algorithms correctly and make sure that each key has "
                "same name as yaml file under '[project_root]/configurations/algorithm' without .yaml suffix"
            )
        return self.compatible_algorithms[algo_name](self.root_cfg.algorithm)

    def exec_task(self, task: str) -> None:
        """
        Executing a certain task specified by string. Each task should be a stage of experiment.
        In most computer vision / nlp applications, tasks should be just train and test.
        In reinforcement learning, you might have more stages such as collecting dataset etc

        Args:
            task: a string specifying a task implemented for this experiment
        """

        if hasattr(self, task) and callable(getattr(self, task)):
            rank_zero_print(cyan("Executing task:"), f"{task} out of {self.cfg.tasks}")
            getattr(self, task)()
        else:
            raise ValueError(
                f"Specified task '{task}' not defined for class {self.__class__.__name__} or is not callable."
            )


class BaseLightningExperiment(BaseExperiment):
    """
    Abstract class for pytorch lightning experiments. Useful for computer vision & nlp where main components are
    simply models, datasets and train loop.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms: Dict = NotImplementedError

    # each key has to be a yaml file under '[project_root]/configurations/dataset' without .yaml suffix
    compatible_datasets: Dict = NotImplementedError
    data_module_cls = BaseDataModule

    def __init__(
        self,
        root_cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        super().__init__(root_cfg, logger, ckpt_path)
        self.data_module = self.data_module_cls(root_cfg, self.compatible_datasets)

    def _build_common_callbacks(self):
        return [EMA(**self.cfg.ema)]

    def training(self) -> None:
        """
        All training happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.training.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))
        if "checkpointing" in self.cfg.training:
            callbacks.append(
                ModelCheckpoint(
                    pathlib.Path(
                        hydra.core.hydra_config.HydraConfig.get()["runtime"][
                            "output_dir"
                        ]
                    )
                    / "checkpoints",
                    **self.cfg.training.checkpointing,
                )
            )
        callbacks += self._build_common_callbacks()

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=(
                DDPStrategy(find_unused_parameters=self.cfg.find_unused_parameters)
                if torch.cuda.device_count() > 1
                else "auto"
            ),
            callbacks=callbacks,
            gradient_clip_val=self.cfg.training.optim.gradient_clip_val,
            val_check_interval=self.cfg.validation.val_every_n_step,
            limit_val_batches=self.cfg.validation.limit_batch,
            check_val_every_n_epoch=self.cfg.validation.val_every_n_epoch,
            accumulate_grad_batches=self.cfg.training.optim.accumulate_grad_batches,
            precision=self.cfg.training.precision,
            detect_anomaly=False,  # self.cfg.debug,
            num_sanity_val_steps=(
                int(self.cfg.debug)
                if self.cfg.validation.num_sanity_val_steps is None
                else self.cfg.validation.num_sanity_val_steps
            ),
            max_epochs=self.cfg.training.max_epochs,
            max_steps=self.cfg.training.max_steps,
            max_time=self.cfg.training.max_time,
            reload_dataloaders_every_n_epochs=self.cfg.reload_dataloaders_every_n_epochs,
        )

        # if self.debug:
        #     self.logger.watch(self.algo, log="all")

        trainer.fit(
            self.algo,
            datamodule=self.data_module,
            ckpt_path=self.ckpt_path,
        )

    def validation(self) -> None:
        """
        All validation happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.validation.compile:
            self.algo = torch.compile(self.algo)

        callbacks = [] + self._build_common_callbacks()
        # import pdb; pdb.set_trace()
        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=(
                DDPStrategy(find_unused_parameters=self.cfg.find_unused_parameters)
                if torch.cuda.device_count() > 1
                else "auto"
            ),
            callbacks=callbacks,
            limit_val_batches=self.cfg.validation.limit_batch,
            precision=self.cfg.validation.precision,
            detect_anomaly=False,  # self.cfg.debug,
            inference_mode=self.cfg.validation.inference_mode,
        )

        # if self.debug:
        #     self.logger.watch(self.algo, log="all")

        trainer.validate(
            self.algo,
            datamodule=self.data_module,
            ckpt_path=self.ckpt_path,
        )

    def test(self) -> None:
        """
        All testing happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.test.compile:
            self.algo = torch.compile(self.algo)

        callbacks = [] + self._build_common_callbacks()

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=(
                DDPStrategy(find_unused_parameters=self.cfg.find_unused_parameters)
                if torch.cuda.device_count() > 1
                else "auto"
            ),
            callbacks=callbacks,
            limit_test_batches=self.cfg.test.limit_batch,
            precision=self.cfg.test.precision,
            detect_anomaly=False,  # self.cfg.debug,
            inference_mode=self.cfg.test.inference_mode,
        )

        # Only load the checkpoint if only testing. Otherwise, it will have been loaded
        # and further trained during train.
        trainer.test(
            self.algo,
            datamodule=self.data_module,
            ckpt_path=self.ckpt_path,
        )
