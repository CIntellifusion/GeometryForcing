"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

import os
import time
from typing import (
    Tuple,
    List,
    TYPE_CHECKING,
    Any,
    Literal,
    Mapping,
    Optional,
    Union,
)
from pathlib import Path
from datetime import timedelta, datetime
from typing_extensions import override
from tqdm import tqdm
from wandb_osh.hooks import TriggerWandbSyncHook
from lightning.pytorch.loggers.wandb import (
    WandbLogger,
    _scan_checkpoints,
    ModelCheckpoint,
    Tensor,
)
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.fabric.utilities.types import _PATH
import wandb
from wandb.apis.public.runs import Run
from utils.print_utils import cyan


if TYPE_CHECKING:
    from wandb.sdk.lib import RunDisabled
    from wandb.wandb_run import Run


class SpaceEfficientWandbLogger(WandbLogger):
    """
    A wandb logger that by default overrides artifacts to save space, instead of creating new version.
    A variable expiration_days can be set to control how long older versions of artifacts are kept.
    By default, the latest version is kept indefinitely, while older versions are kept for 1 days.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = False,
        dir: Optional[_PATH] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        experiment: Union["Run", "RunDisabled", None] = None,
        prefix: str = "",
        checkpoint_name: Optional[str] = None,
        expiration_days: Optional[int] = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=False,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )

        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=offline,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )
        self.expiration_days = expiration_days
        self._last_artifacts = []

    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        print("[Wandb Warrning By Haoyu]: we pass this function to avoid crash when uploading checkpoints. The checkpionts are saved to blob")
        return 
        import wandb

        # get checkpoints to be saved with associated score
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

        # log iteratively all new checkpoints
        artifacts = []
        for t, p, s, tag in checkpoints:
            metadata = {
                "score": s.item() if isinstance(s, Tensor) else s,
                "original_filename": Path(p).name,
                checkpoint_callback.__class__.__name__: {
                    k: getattr(checkpoint_callback, k)
                    for k in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                    ]
                    # ensure it does not break if `ModelCheckpoint` args change
                    if hasattr(checkpoint_callback, k)
                },
            }
            if not self._checkpoint_name:
                self._checkpoint_name = f"model-{self.experiment.id}"

            artifact = wandb.Artifact(
                name=self._checkpoint_name, type="model", metadata=metadata
            )
            artifact.add_file(p, name="model.ckpt")
            aliases = (
                ["latest", "best"]
                if p == checkpoint_callback.best_model_path
                else ["latest"]
            )
            self.experiment.log_artifact(artifact, aliases=aliases)
            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[p] = t
            artifacts.append(artifact)

        for artifact in self._last_artifacts:
            if not self._offline:
                artifact.wait()
            artifact.ttl = timedelta(days=self.expiration_days)
            artifact.save()
        self._last_artifacts = artifacts


class OfflineWandbLogger(SpaceEfficientWandbLogger):
    """
    Wraps WandbLogger to trigger offline sync hook occasionally.
    This is useful when running on slurm clusters, many of which
    only has internet on login nodes, not compute nodes.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: _PATH = ".",
        version: Optional[str] = None,
        offline: bool = False,
        dir: Optional[_PATH] = None,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        project: Optional[str] = None,
        log_model: Union[Literal["all"], bool] = False,
        experiment: Union["Run", "RunDisabled", None] = None,
        prefix: str = "",
        checkpoint_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            save_dir=save_dir,
            version=version,
            offline=False,
            dir=dir,
            id=id,
            anonymous=anonymous,
            project=project,
            log_model=log_model,
            experiment=experiment,
            prefix=prefix,
            checkpoint_name=checkpoint_name,
            **kwargs,
        )
        self._offline = offline
        communication_dir = Path(".wandb_osh_command_dir")
        communication_dir.mkdir(parents=True, exist_ok=True)
        self.trigger_sync = TriggerWandbSyncHook(communication_dir)
        self.last_sync_time = 0.0
        self.min_sync_interval = 60
        self.wandb_dir = os.path.join(self._save_dir, "wandb/latest-run")

    @override
    @rank_zero_only
    def log_metrics(
        self, metrics: Mapping[str, float], step: Optional[int] = None
    ) -> None:
        out = super().log_metrics(metrics, step)
        if time.time() - self.last_sync_time > self.min_sync_interval:
            self.trigger_sync(self.wandb_dir)
            self.last_sync_time = time.time()
        return out


def cleanup_project(
    entity: str,
    project: str,
    log_folder: Optional[str] = None,
    ignore_ttl: bool = False,
):
    """
    cleanup the project by applying TTL policy to the model artifacts
    """
    num_deleted = 0
    total_size = 0
    log_file = Path(log_folder) / f"{datetime.now().strftime('%Y-%m-%d')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as f:
        f.write(f"[Cleanup] {entity}/{project}\n\n")
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", order="-created_at")
    tbar = tqdm(runs)
    for run in tbar:
        versions, size = cleanup_run(run, ignore_ttl)
        num_deleted += len(versions)
        total_size += size
        tbar.set_postfix(
            num_deleted=num_deleted,
            saved=f"{total_size:.2f} GB",
        )
        if len(versions) > 0:
            with open(log_file, "a") as f:
                f.write(f"{run.id}\n{run.name}\n{versions}\n{size:.2f} GB\n\n")

    print(cyan(f"Deleted {num_deleted} models, saved {total_size:.2f} GB"))


def cleanup_run(run: Run, ignore_ttl: bool = False) -> Tuple[List[str], float]:
    """
    cleanup the models that are not best or latest and have expired
    Returns: size of the deleted artifacts (in GB)
    """
    size = 0
    versions = []
    for artifact in run.logged_artifacts():
        if (
            artifact.type == "model"
            and artifact.state == "COMMITTED"
            and (
                "best" not in artifact.aliases
                and "latest" not in artifact.aliases
                and "backup" not in artifact.aliases
            )
            and (artifact.ttl is not None or ignore_ttl)
        ):
            should_delete = True
            if not ignore_ttl:
                created_at = datetime.strptime(
                    artifact.created_at, "%Y-%m-%dT%H:%M:%SZ"
                )
                current_time = datetime.now()
                should_delete = current_time - created_at > artifact.ttl
            if should_delete:
                versions.append(artifact.version)
                size += artifact.size / 1024**3
                artifact.delete()
    return versions, size


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ignore-ttl",
        action="store_true",
        help="Ignore TTL policy and delete non-best, non-latest models",
    )
    args = parser.parse_args()
    cleanup_project(
        "scene-representation-group",
        "video_diffusion",
        "wandb_cleanup",
        ignore_ttl=args.ignore_ttl,
    )
