from typing import Optional, Any, Dict, Literal, Callable, Tuple
from functools import partial
from omegaconf import DictConfig
import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
from torch import Tensor
from torch.optim.optimizer import Optimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.utilities import grad_norm
from einops import rearrange, repeat, reduce
from transformers import get_scheduler
from tqdm import tqdm
from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from algorithms.common.metrics.video import VideoMetric, SharedVideoMetricModelRegistry
from algorithms.vae import ImageVAE, VideoVAE
from utils.print_utils import cyan
from utils.distributed_utils import rank_zero_print, is_rank_zero
from utils.torch_utils import bernoulli_tensor
from utils.logging_utils import log_video
from utils.torch_utils import freeze_model
from .diffusion import (
    DiscreteDiffusion,
    ContinuousDiffusion,
)
from .history_guidance import HistoryGuidance


class DFoTVideo(BasePytorchAlgo):
    """
    An algorithm for training and evaluating
    Diffusion Forcing Transformer (DFoT) for video generation.
    """

    def __init__(self, cfg: DictConfig):
        # 1. Shape
        self.x_shape = list(cfg.x_shape)
        self.frame_skip = cfg.frame_skip
        self.chunk_size = cfg.chunk_size
        self.external_cond_dim = cfg.external_cond_dim * (
            cfg.frame_skip if cfg.external_cond_stack else 1
        )

        # 2. Latent
        self.is_latent_diffusion = cfg.latent.enable
        self.is_latent_online = cfg.latent.type == "online"
        self.temporal_downsampling_factor = cfg.latent.downsampling_factor[0]
        self.is_latent_video_vae = self.temporal_downsampling_factor > 1
        if self.is_latent_diffusion:
            self.x_shape = [cfg.latent.num_channels] + [
                d // cfg.latent.downsampling_factor[1] for d in self.x_shape[1:]
            ]
            if self.is_latent_video_vae:
                self.check_video_vae_compatibility(cfg)

        # 3. Diffusion
        self.use_causal_mask = cfg.diffusion.use_causal_mask
        self.timesteps = cfg.diffusion.timesteps
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.clip_noise = cfg.diffusion.clip_noise
        if "cum_snr_decay" in cfg.diffusion.loss_weighting:
            cfg.diffusion.loss_weighting.cum_snr_decay = (
                cfg.diffusion.loss_weighting.cum_snr_decay**cfg.frame_skip
            )
        self.is_full_sequence = (
            cfg.noise_level == "random_uniform"
            and not cfg.fixed_context.enabled
            and not cfg.variable_context.enabled
        )

        # 4. Logging
        self.logging = cfg.logging
        self.tasks = [
            task
            for task in ["prediction", "interpolation"]
            if getattr(cfg.tasks, task).enabled
        ]
        self.num_logged_videos = 0
        self.generator = None

        super().__init__(cfg)

    # ---------------------------------------------------------------------
    # Prepare Model, Optimizer, and Metrics
    # ---------------------------------------------------------------------

    def _build_model(self):
        # 1. Diffusion model
        diffusion_cls = (
            ContinuousDiffusion
            if self.cfg.diffusion.is_continuous
            else DiscreteDiffusion
        )

        if self.cfg.compile:
            # NOTE: this compiling is only for training speedup
            if self.cfg.compile == "true_without_ddp_optimizer":
                # NOTE: `cfg.compile` should be set to this value when using `torch.compile` with DDP & Gradient Checkpointing
                # Otherwise, torch.compile will raise an error.
                # Reference: https://github.com/pytorch/pytorch/issues/104674
                # pylint: disable=protected-access
                torch._dynamo.config.optimize_ddp = False
            assert (
                self.cfg.diffusion.is_continuous
            ), "`torch.compile` is only verified for continuous-time diffusion models. To use it for discrete-time models, it should be tested, including # graph breaks"
        if "controlnet" in self.cfg._name:
            self.cfg.compile = False
            print(f"{cyan('torch.compile is not supported for ControlNet models, disabling it')}")
            
        self.diffusion_model = torch.compile(
            diffusion_cls(
                cfg=self.cfg.diffusion,
                backbone_cfg=self.cfg.backbone,
                x_shape=self.x_shape,
                max_tokens=self.max_tokens,
                external_cond_dim=self.external_cond_dim,
            ),
            disable=not self.cfg.compile,
        )
        # import pdb; pdb.set_trace()
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)

        # 2. VAE model
        if self.is_latent_diffusion and self.is_latent_online:
            self._load_vae()
        else:
            self.vae = None

        # 3. Metrics
        if len(self.tasks) == 0:
            return
        registry = SharedVideoMetricModelRegistry()
        metric_types = self.logging.metrics
        for task in self.tasks:
            match task:
                case "prediction":
                    self.metrics_prediction = VideoMetric(
                        registry,
                        metric_types,
                        split_batch_size=self.logging.metrics_batch_size,
                    )
                case "interpolation":
                    assert (
                        not self.use_causal_mask
                        and not self.is_full_sequence
                        and self.max_tokens > 2
                    ), "To execute interpolation, the model must be non-causal, not full sequence, and be able to process more than 2 tokens."
                    self.metrics_interpolation = VideoMetric(
                        registry,
                        metric_types,
                        split_batch_size=self.logging.metrics_batch_size,
                    )

    def configure_optimizers(self):
        transition_params = list(self.diffusion_model.parameters())
        optimizer_dynamics = torch.optim.AdamW(
            transition_params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.optimizer_beta,
        )

        lr_scheduler_config = {
            "scheduler": get_scheduler(
                optimizer=optimizer_dynamics,
                **self.cfg.lr_scheduler,
            ),
            "interval": "step",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer_dynamics,
            "lr_scheduler": lr_scheduler_config,
        }

    def _load_vae(self) -> None:
        """
        Load the pretrained VAE model.
        """
        vae_cls = VideoVAE if self.is_latent_video_vae else ImageVAE
        self.vae = vae_cls.from_pretrained(
            path=self.cfg.vae.pretrained_path,
            torch_dtype=(
                torch.float16 if self.cfg.vae.use_fp16 else torch.float32
            ),  # only for Diffuser's ImageVAE
            **self.cfg.vae.pretrained_kwargs,
        ).to(self.device)
        freeze_model(self.vae)

    def _metrics(
        self,
        task: Literal["prediction", "interpolation"],
    ) -> Optional[VideoMetric]:
        """
        Get the appropriate metrics object for the given task.
        """
        return getattr(self, f"metrics_{task}", None)

    # ---------------------------------------------------------------------
    # Length-related Properties and Utils
    # NOTE: "Frame" and "Token" should be distinguished carefully.
    # "Frame" refers to original unit of data loaded from dataset.
    # "Token" refers to the unit of data processed by the diffusion model.
    # The two differ when using a VAE for latent diffusion.
    # ---------------------------------------------------------------------

    def _n_frames_to_n_tokens(self, n_frames: int) -> int:
        """
        Converts the number of frames to the number of tokens.
        - Chunk-wise VideoVAE: 1st frame -> 1st token, then every self.temporal_downsampling_factor frames -> next token.
        - ImageVAE or Non-latent Diffusion: 1 token per frame.
        """
        return (n_frames - 1) // self.temporal_downsampling_factor + 1

    def _n_tokens_to_n_frames(self, n_tokens: int) -> int:
        """
        Converts the number of tokens to the number of frames.
        """
        return (n_tokens - 1) * self.temporal_downsampling_factor + 1

    # ---------------------------------------------------------------------
    # NOTE: max_{frames, tokens} indicates the maximum number of frames/tokens
    # that the model can process within a single forward pass.
    # ---------------------------------------------------------------------

    @property
    def max_frames(self) -> int:
        return self.cfg.max_frames

    @property
    def max_tokens(self) -> int:
        return self._n_frames_to_n_tokens(self.max_frames)

    # ---------------------------------------------------------------------
    # NOTE: n_{frames, tokens} indicates the number of frames/tokens
    # that the model actually processes during training/validation.
    # During validation, it may be different from max_{frames, tokens},
    # ---------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        return self.max_frames if self.trainer.training else self.cfg.n_frames

    @property
    def n_context_frames(self) -> int:
        return self.cfg.context_frames

    @property
    def n_tokens(self) -> int:
        return self._n_frames_to_n_tokens(self.n_frames)

    @property
    def n_context_tokens(self) -> int:
        return self._n_frames_to_n_tokens(self.n_context_frames)

    # ---------------------------------------------------------------------
    # Data Preprocessing
    # ---------------------------------------------------------------------

    def on_after_batch_transfer(
        self, batch: Dict, dataloader_idx: int
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
        """
        Preprocess the batch before training/validation.

        Args:
            batch (Dict): The batch of data. Contains "videos" or "latents", (optional) "conditions", and "masks".
            dataloader_idx (int): The index of the dataloader.
        Returns:
            xs (Tensor, "B n_tokens *x_shape"): Tokens to be processed by the model.
            conditions (Optional[Tensor], "B n_tokens d"): External conditions for the tokens.
            masks (Tensor, "B n_tokens"): Masks for the tokens.
            gt_videos (Optional[Tensor], "B n_frames *x_shape"): Optional ground truth videos, used for validation in latent diffusion.
        """
        # 1. Tokenize the videos and optionally prepare the ground truth videos
        gt_videos = None
        if self.is_latent_diffusion:
            xs = (
                self._encode(batch["videos"])
                if self.is_latent_online
                else batch["latents"]
            )
            if "videos" in batch:
                gt_videos = batch["videos"]
        else:
            xs = batch["videos"]
            gt_videos = batch.get("videos", None) 
        # import pdb; pdb.set_trace() 
        xs = self._normalize_x(xs)

        # 2. Prepare external conditions
        conditions = batch.get("conds", None)

        # 3. Prepare the masks
        if "masks" in batch:
            assert (
                not self.is_latent_video_vae
            ), "Masks should not be provided from the dataset when using VideoVAE."
        else:
            masks = torch.ones(*xs.shape[:2]).bool().to(self.device)

        return xs, conditions, masks, gt_videos

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------

    def training_step(self, batch, batch_idx, namespace="training") -> STEP_OUTPUT:
        """Training step"""
        xs, conditions, masks, *_ = batch

        noise_levels, masks = self._get_training_noise_levels(xs, masks)
        xs_pred, loss = self.diffusion_model(
            xs,
            self._process_conditions(conditions),
            k=noise_levels,
        )
        loss = self._reweight_loss(loss, masks)

        if batch_idx % self.cfg.logging.loss_freq == 0:
            self.log(
                f"{namespace}/loss",
                loss,
                on_step=namespace == "training",
                on_epoch=namespace != "training",
                sync_dist=True,
            )

        xs, xs_pred = map(self._unnormalize_x, (xs, xs_pred))

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }
        
        return output_dict

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        if (
            self.cfg.logging.grad_norm_freq
            and self.global_step % self.cfg.logging.grad_norm_freq == 0
        ):
            norms = grad_norm(self.diffusion_model, norm_type=2)
            # NOTE: `norms` need not be gathered, as they are already uniform across all devices
            self.log_dict(norms)

    # ---------------------------------------------------------------------
    # Validation & Test
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        """Validation step"""
        # 1. If running validation while training a model, directly evaluate
        # the denoising performance to detect overfitting, etc.
        # Logs the "denoising_vis" visualization as well as "validation/loss" metric.
        # print("short cut")
        # return 
        if self.trainer.state.fn == "FIT":
            self._eval_denoising(batch, batch_idx, namespace=namespace)
        # import pdb; pdb.set_trace()
        # 2. Sample all videos (based on the specified tasks)
        # and log the generated videos and metrics.
        if not (
            self.trainer.sanity_checking and not self.cfg.logging.sanity_generation
        ):
            all_videos = self._sample_all_videos(batch, batch_idx, namespace)
            self._update_metrics(all_videos)
            self._log_videos(all_videos, namespace)
    
    def on_validation_epoch_start(self) -> None:
        torch.cuda.empty_cache()
        if self.cfg.logging.deterministic is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(
                self.global_rank
                + self.trainer.world_size * self.cfg.logging.deterministic
            )
        if self.is_latent_diffusion and not self.is_latent_online:
            self._load_vae()
    def on_validation_epoch_end(self, namespace="validation") -> None:
        self.generator = None
        if self.is_latent_diffusion and not self.is_latent_online:
            self.vae = None
        self.num_logged_videos = 0

        if self.trainer.sanity_checking and not self.cfg.logging.sanity_generation:
            return

        for task in self.tasks:
            self.log_dict(
                self._metrics(task).log(task),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        torch.cuda.empty_cache()

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.validation_step(*args, **kwargs, namespace="test")

    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end(namespace="test")

    # ---------------------------------------------------------------------
    # Denoising Evaluation
    # ---------------------------------------------------------------------

    def _eval_denoising(self, batch, batch_idx, namespace="training") -> None:
        """Evaluate the denoising performance during training."""
        xs, conditions, masks, gt_videos = batch

        xs = xs[:, : self.max_tokens]
        if conditions is not None:
            conditions = conditions[:, : self.max_tokens]
        masks = masks[:, : self.max_tokens]
        if gt_videos is not None:
            gt_videos = gt_videos[:, : self.max_frames]

        batch = (xs, conditions, masks, gt_videos)
        output = self.training_step(batch, batch_idx, namespace=namespace)

        gt_videos = gt_videos if self.is_latent_diffusion else output["xs"]
        recons = output["xs_pred"]
        if self.is_latent_diffusion:
            recons = self._decode(recons)

        if recons.shape[1] < gt_videos.shape[1]:  # recons.ndim is 5
            recons = F.pad(
                recons,
                (0, 0, 0, 0, 0, 0, 0, gt_videos.shape[1] - recons.shape[1], 0, 0),
            )

        gt_videos, recons = self.gather_data((gt_videos, recons))

        if not (
            is_rank_zero
            and self.logger
            and self.num_logged_videos < self.logging.max_num_videos
        ):
            return

        num_videos_to_log = min(
            self.logging.max_num_videos - self.num_logged_videos,
            gt_videos.shape[0],
        )
        log_video(
            recons[:num_videos_to_log],
            gt_videos[:num_videos_to_log],
            step=self.global_step,
            namespace="denoising_vis",
            logger=self.logger.experiment,
            indent=self.num_logged_videos,
            captions="denoised | gt",
        )

    # ---------------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------------

    def _sample_all_videos(
        self, batch, batch_idx, namespace="validation"
    ) -> Optional[Dict[str, Tensor]]:
        xs, conditions, *_, gt_videos = batch
        all_videos: Dict[str, Tensor] = {"gt": xs}

        for task in self.tasks:
            sample_fn = (
                self._predict_videos
                if task == "prediction"
                else self._interpolate_videos
            )
            all_videos[task] = sample_fn(xs, conditions=conditions)
        # import pdb; pdb.set_trace()
        # remove None values
        all_videos = {k: v for k, v in all_videos.items() if v is not None}
        # rearrange/unnormalize/detach the videos
        all_videos = {k: self._unnormalize_x(v).detach() for k, v in all_videos.items()}
        # decode latents if using latents
        if self.is_latent_diffusion:
            all_videos = {
                k: self._decode(v) if k != "gt" else gt_videos
                for k, v in all_videos.items()
            }

        # # replace the context frames of video predictions with the ground truth
        if "prediction" in all_videos:
            all_videos["prediction"][:, : self.n_context_frames] = all_videos["gt"][
                :, : self.n_context_frames
            ]

        return all_videos

    def _predict_videos(
        self, xs: Tensor, conditions: Optional[Tensor] = None
    ) -> Tensor:
        """
        Predict the videos with the given context, using sliding window rollouts if necessary.
        Optionally, if cfg.tasks.prediction.keyframe_density < 1, predict the keyframes first,
        then interpolate the missing intermediate frames.
        """
        xs_pred = xs.clone()

        history_guidance = HistoryGuidance.from_config(
            config=self.cfg.tasks.prediction.history_guidance,
            timesteps=self.timesteps,
        )

        density = self.cfg.tasks.prediction.keyframe_density or 1
        if density > 1:
            raise ValueError("tasks.prediction.keyframe_density must be <= 1")
        keyframe_indices = (
            torch.linspace(0, xs_pred.shape[1] - 1, round(density * xs_pred.shape[1]))
            .round()
            .long()
        )
        keyframe_indices = torch.cat(
            [torch.arange(self.n_context_tokens), keyframe_indices]
        ).unique()  # context frames are always keyframes
        key_conditions = (
            conditions[:, keyframe_indices] if conditions is not None else None
        )
        # import pdb; pdb.set_trace()
        # 1. Predict the keyframes
        xs_pred_key, *_ = self._predict_sequence(
            xs_pred[:, : self.n_context_tokens],
            length=len(keyframe_indices),
            conditions=key_conditions,
            history_guidance=history_guidance,
            reconstruction_guidance=self.cfg.diffusion.reconstruction_guidance,
            sliding_context_len=self.cfg.tasks.prediction.sliding_context_len
            or self.max_tokens // 2,
        )
        xs_pred[:, keyframe_indices] = xs_pred_key
        # if is_rank_zero: # uncomment to visualize history guidance
        #     history_guidance.log(logger=self.logger)

        # 2. (Optional) Interpolate the intermediate frames
        if len(keyframe_indices) < xs_pred.shape[1]:
            context_mask = torch.zeros(xs_pred.shape[:2], device=self.device).bool()
            context_mask[:, keyframe_indices] = True
            xs_pred = self._interpolate_videos(
                context=xs_pred,
                context_mask=context_mask,
                conditions=conditions,
            )

        return xs_pred

    def _pad_to_max_tokens(self, y: Optional[Tensor]) -> Tensor:
        """Given a tensor y of shape (B, T, ...), pad it at the end across the time dimension to have a length of self.max_tokens."""
        if y is None:
            return y
        if y.shape[1] < self.max_tokens:
            y = torch.cat(
                [
                    y,
                    repeat(
                        y[:, -1:],
                        "b 1 ... -> b t ...",
                        t=self.max_tokens - y.shape[1],
                    ),
                ],
                dim=1,
            )
        return y

    def _interpolate_videos(
        self,
        context: Tensor,
        context_mask: Optional[Tensor] = None,
        conditions: Optional[Tensor] = None,
    ) -> Tensor:
        """
        A general method for frame interpolation. Given a video of any length > 2, when the left and right key frames are known, it (iteratively, if necessary) interpolates the video, filling out all missing frames.

        The logic is as follows:
        1. If the distance between adjacent key frames >= self.max_tokens - 1, it will first infer equally spaced self.max_tokens - 2 frames between the key frames.
        2. Otherwise, it will increase the number of key frames until right before the distance between adjacent key frames > self.max_tokens - 1, then pad the video with with the last key frame (to keep the model input size self.max_tokens).
        3. Repeat the above process until all missing frames are filled.

        Args:
            context (Tensor, "B T C H W"): The video including the context frames.
            context_mask (Optional[Tensor], "B T"): The mask for the context frames. True for the context frames, False otherwise. If None, *only the first and last frames* are considered as key frames. It is assumed that context_mask is identical for all videos in the batch, as "interpolation plan" depends on the context_mask.
            conditions (Optional[Tensor], "B T ..."): The external conditions for the video.
            history_guidance (Optional[HistoryGuidance]): The history guidance object - if None, it will be initialized from the config.
        """
        # Generate default context mask if not provided
        if context_mask is None:
            context_mask = torch.zeros(
                context.shape[0], context.shape[1], device=self.device
            ).bool()
            context_mask[:, [0, -1]] = True
        else:
            assert context_mask[
                :, [0, -1]
            ].all(), "The first and last frames must be known to interpolate."

        # enable using different history guidance scheme for interpolation
        history_guidance = HistoryGuidance.from_config(
            config=self.cfg.tasks.interpolation.history_guidance,
            timesteps=self.timesteps,
        )

        # Generate a plan for frame interpolation
        plan = []
        plan_mask = context_mask[0].clone()
        while not plan_mask.all():
            key_frames = torch.where(plan_mask)[0]
            current_plan = []  # plan for the current iteration
            current_chunk = None  # chunk to be merged with the next chunk
            for left, right in zip(key_frames[:-1], key_frames[1:]):
                if current_chunk is not None:
                    if (
                        len(current_chunk) + right - left <= self.max_tokens
                    ):  # merge with the next chunk if possible
                        current_chunk = torch.cat(
                            [
                                current_chunk,
                                torch.arange(
                                    left + 1,
                                    right + 1,
                                    device=self.device,
                                ),
                            ]
                        )
                        continue
                    # if cannot merge, add the current chunk to the plan
                    current_plan.append(current_chunk)
                    current_chunk = None

                if right - left == 1:  # no missing frames
                    continue

                if right - left >= self.max_tokens - 1:  # Case 1
                    current_plan.append(
                        torch.linspace(left, right, self.max_tokens, device=self.device)
                        .round()
                        .long()
                    )
                else:  # Case 2
                    current_chunk = torch.arange(left, right + 1, device=self.device)
            if current_chunk is not None:
                current_plan.append(current_chunk)
            for frames in current_plan:
                plan_mask[frames] = True
            plan.append(current_plan)

        # Execute the plan
        xs = context.clone()
        context_mask = context_mask.clone()
        max_batch_size = self.cfg.tasks.interpolation.max_batch_size
        pbar = tqdm(
            total=sum(
                [
                    (
                        (len(frames) + max_batch_size - 1) // max_batch_size
                        if max_batch_size
                        else 1
                    )
                    for frames in plan
                ]
            )
            * self.sampling_timesteps,
            initial=0,
            desc="Interpolating with DFoT",
            leave=False,
            disable=False,
        )
        # import pdb; pdb.set_trace()
        for current_plan in plan:
            # Collect the batched input for the current plan
            current_context = []
            current_context_mask = []
            current_conditions = [] if conditions is not None else None
            for frames in current_plan:
                current_context.append(self._pad_to_max_tokens(xs[:, frames]))
                current_context_mask.append(
                    self._pad_to_max_tokens(context_mask[:, frames])
                )
                if conditions is not None:
                    current_conditions.append(
                        self._pad_to_max_tokens(conditions[:, frames])
                    )
            current_context, current_context_mask, current_conditions = map(
                lambda y: torch.cat(y, 0) if y is not None else None,
                (current_context, current_context_mask, current_conditions),
            )
            xs_pred = []
            # Interpolate the video in parallel,
            # while keeping the batch size smaller than the maximum batch size to avoid memory errors
            max_batch_size = (
                self.cfg.tasks.interpolation.max_batch_size or current_context.shape[0]
            )
            for (
                current_context_chunk,
                current_context_mask_chunk,
                current_conditions_chunk,
            ) in zip(
                current_context.split(max_batch_size, 0),
                current_context_mask.split(max_batch_size, 0),
                (
                    current_conditions.split(max_batch_size, 0)
                    if current_conditions is not None
                    else [None] * (current_context.shape[0] // max_batch_size)
                ),
            ):
                batch_size = current_context_chunk.shape[0]
                # import pdb; pdb.set_trace()
                xs_pred_chunk, _ = self._sample_sequence(
                    batch_size=batch_size,
                    context=current_context_chunk,
                    context_mask=current_context_mask_chunk.long(),
                    conditions=current_conditions_chunk,
                    history_guidance=history_guidance,
                    pbar=pbar,
                )
                xs_pred.append(xs_pred_chunk)
            xs_pred = torch.cat(xs_pred, 0)
            # Update with the interpolated frames
            for frames, pred in zip(current_plan, xs_pred.chunk(len(current_plan), 0)):
                xs[:, frames] = pred[:, : len(frames)]
                context_mask[:, frames] = True
        pbar.close()
        return xs

    # ---------------------------------------------------------------------
    # Logging (Metrics, Videos)
    # ---------------------------------------------------------------------

    def _update_metrics(self, all_videos: Dict[str, Tensor]) -> None:
        """Update all metrics during validation/test step."""
        if (
            self.logging.n_metrics_frames is not None
        ):  # only consider the first n_metrics_frames for evaluation
            all_videos = {
                k: v[:, : self.logging.n_metrics_frames] for k, v in all_videos.items()
            }

        gt_videos = all_videos["gt"]
        for task in self.tasks:
            metric = self._metrics(task)
            videos = all_videos[task]
            context_mask = torch.zeros(self.n_frames).bool().to(self.device)
            match task:
                case "prediction":
                    context_mask[: self.n_context_frames] = True
                case "interpolation":
                    context_mask[[0, -1]] = True
            if self.logging.n_metrics_frames is not None:
                context_mask = context_mask[: self.logging.n_metrics_frames]
            metric(videos, gt_videos, context_mask=context_mask)

    def _log_videos(self, all_videos: Dict[str, Tensor], namespace: str) -> None:
        """Log videos during validation/test step."""
        all_videos = self.gather_data(all_videos)
        batch_size, n_frames = all_videos["gt"].shape[:2]

        if not (
            is_rank_zero
            and self.logger
            and self.num_logged_videos < self.logging.max_num_videos
        ):
            return

        num_videos_to_log = min(
            self.logging.max_num_videos - self.num_logged_videos,
            batch_size,
        )
        cut_videos = lambda x: x[:num_videos_to_log]

        for task in self.tasks:
            log_video(
                cut_videos(all_videos[task]),
                cut_videos(all_videos["gt"]),
                step=None if namespace == "test" else self.global_step,
                namespace=f"{task}_vis",
                logger=self.logger.experiment,
                indent=self.num_logged_videos,
                raw_dir=self.logging.raw_dir,
                context_frames=(
                    self.n_context_frames
                    if task == "prediction"
                    else torch.tensor(
                        [0, n_frames - 1], device=self.device, dtype=torch.long
                    )
                ),
                captions=f"{task} | gt",
            )

        self.num_logged_videos += batch_size

    # ---------------------------------------------------------------------
    # Data Preprocessing Utils
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def _process_conditions(
        self,
        conditions: Optional[Tensor],
        noise_levels: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """
        Post-process the conditions before feeding them to the model.
        For example, conditions that should be computed relatively (e.g. relative poses)
        should be processed here instead of the dataset.

        Args:
            conditions (Optional[Tensor], "B T ..."): The external conditions for the video.
            noise_levels (Optional[Tensor], "B T"): Current noise levels for each token during sampling
        """

        if conditions is None:
            return conditions
        match self.cfg.external_cond_processing:
            case "mask_first":
                mask = torch.ones_like(conditions)
                mask[:, :1, : self.external_cond_dim] = 0
                return conditions * mask
            case _:
                raise NotImplementedError(
                    f"External condition processing {self.cfg.external_cond_processing} is not implemented."
                )

    # ---------------------------------------------------------------------
    # Training Utils
    # ---------------------------------------------------------------------

    def _get_training_noise_levels(
        self, xs: Tensor, masks: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """Generate random noise levels for training."""
        batch_size, n_tokens, *_ = xs.shape

        # random function different for continuous and discrete diffusion
        rand_fn = partial(
            *(
                (torch.rand,)
                if self.cfg.diffusion.is_continuous
                else (torch.randint, 0, self.timesteps)
            ),
            device=xs.device,
            generator=self.generator,
        )

        # baseline training (SD: fixed_context, BD: variable_context)
        context_mask = None
        if self.cfg.variable_context.enabled:
            assert (
                not self.cfg.fixed_context.enabled
            ), "Cannot use both fixed and variable context"
            context_mask = bernoulli_tensor(
                (batch_size, n_tokens),
                self.cfg.variable_context.prob,
                device=self.device,
                generator=self.generator,
            ).bool()
        elif self.cfg.fixed_context.enabled:
            context_indices = self.cfg.fixed_context.indices or list(
                range(self.n_context_tokens)
            )
            context_mask = torch.zeros(
                (batch_size, n_tokens), dtype=torch.bool, device=xs.device
            )
            context_mask[:, context_indices] = True

        match self.cfg.noise_level:
            case "random_independent":  # independent noise levels (Diffusion Forcing)
                noise_levels = rand_fn((batch_size, n_tokens))
            case "random_uniform":  # uniform noise levels (Typical Video Diffusion)
                noise_levels = rand_fn((batch_size, 1)).repeat(1, n_tokens)

        if self.cfg.uniform_future.enabled:  # simplified training (Appendix A.5)
            noise_levels[:, self.n_context_tokens :] = rand_fn((batch_size, 1)).repeat(
                1, n_tokens - self.n_context_tokens
            )

        # treat frames that are not available as "full noise"
        noise_levels = torch.where(
            reduce(masks.bool(), "b t ... -> b t", torch.any),
            noise_levels,
            torch.full_like(
                noise_levels,
                1 if self.cfg.diffusion.is_continuous else self.timesteps - 1,
            ),
        )

        if context_mask is not None:
            # binary dropout training to enable guidance
            dropout = (
                (
                    self.cfg.variable_context
                    if self.cfg.variable_context.enabled
                    else self.cfg.fixed_context
                ).dropout
                if self.trainer.training
                else 0.0
            )
            context_noise_levels = bernoulli_tensor(
                (batch_size, 1),
                dropout,
                device=xs.device,
                generator=self.generator,
            )
            if not self.cfg.diffusion.is_continuous:
                context_noise_levels = context_noise_levels.long() * (
                    self.timesteps - 1
                )
            noise_levels = torch.where(context_mask, context_noise_levels, noise_levels)

            # modify masks to exclude context frames from loss computation
            context_mask = rearrange(
                context_mask, "b t -> b t" + " 1" * len(masks.shape[2:])
            )
            masks = torch.where(context_mask, False, masks)

        return noise_levels, masks

    def _reweight_loss(self, loss, weight=None):
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape)
            weight = rearrange(
                weight,
                "... -> ..." + " 1" * expand_dim,
            )
            loss = loss * weight

        return loss.mean()

    # ---------------------------------------------------------------------
    # Sampling Utils
    # ---------------------------------------------------------------------

    def _generate_scheduling_matrix(
        self,
        horizon: int,
        padding: int = 0,
    ):
        match self.cfg.scheduling_matrix:
            case "full_sequence":
                scheduling_matrix = np.arange(self.sampling_timesteps, -1, -1)[
                    :, None
                ].repeat(horizon, axis=1)
            case "autoregressive":
                scheduling_matrix = self._generate_pyramid_scheduling_matrix(
                    horizon, self.sampling_timesteps
                )

        scheduling_matrix = torch.from_numpy(scheduling_matrix).long()

        scheduling_matrix = self.diffusion_model.ddim_idx_to_noise_level(
            scheduling_matrix
        )

        # paded entries are labeled as pure noise
        scheduling_matrix = F.pad(
            scheduling_matrix, (0, padding, 0, 0), value=self.timesteps - 1
        )

        return scheduling_matrix

    def _predict_sequence(
        self,
        context: torch.Tensor,
        length: Optional[int] = None,
        conditions: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        reconstruction_guidance: float = 0.0,
        history_guidance: Optional[HistoryGuidance] = None,
        sliding_context_len: Optional[int] = None,
        return_all: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict a sequence given context tokens at the beginning, using sliding window if necessary.
        Args
        ----
        context: torch.Tensor, Shape (batch_size, init_context_len, *self.x_shape)
            Initial context tokens to condition on
        length: Optional[int]
            Desired number of tokens in sampled sequence.
            If None, fall back to to self.max_tokens, and
            If bigger than self.max_tokens, sliding window sampling will be used.
        conditions: Optional[torch.Tensor], Shape (batch_size, conditions_len, ...)
            Unprocessed external conditions for sampling, e.g. action or text, optional
        guidance_fn: Optional[Callable]
            Guidance function for sampling
        reconstruction_guidance: float
            Scale of reconstruction guidance (from Video Diffusion Models Ho. et al.)
        history_guidance: Optional[HistoryGuidance]
            History guidance object that handles compositional generation
        sliding_context_len: Optional[int]
            Max context length when using sliding window. -1 to use max_tokens - 1.
            Has no influence when length <= self.max_tokens as no sliding window is needed.
        return_all: bool
            Whether to return all steps of the sampling process.

        Returns
        -------
        xs_pred: torch.Tensor, Shape (batch_size, length, *self.x_shape)
            Predicted sequence with both context and generated tokens
        record: Optional[torch.Tensor], Shape (num_steps, batch_size, length, *self.x_shape)
            Record of all steps of the sampling process
        """
        if length is None:
            length = self.max_tokens
        if sliding_context_len is None:
            if self.max_tokens < length:
                raise ValueError(
                    "when length > max_tokens, sliding_context_len must be specified."
                )
            else:
                sliding_context_len = self.max_tokens - 1
        if sliding_context_len == -1:
            sliding_context_len = self.max_tokens - 1
            
        if context is not None and context.isnan().any():
            raise ValueError(
                "NaN values detected in the context. "
                "Check the model and the input data."
            )
        batch_size, gt_len, *_ = context.shape

        if sliding_context_len < gt_len:
            raise ValueError(
                "sliding_context_len is expected to be >= length of initial context,"
                f"got {sliding_context_len}. If you are trying to use max context, "
                "consider specifying sliding_context_len=-1."
            )

        chunk_size = self.chunk_size if self.use_causal_mask else self.max_tokens

        curr_token = gt_len
        xs_pred = context
        x_shape = self.x_shape
        record = None
        pbar = tqdm(
            total=self.sampling_timesteps
            * (
                1
                + (length - sliding_context_len - 1)
                // (self.max_tokens - sliding_context_len)
            ),
            initial=0,
            desc="Predicting with DFoT",
            leave=False,
            disable=False
        )
        while curr_token < length:
            if record is not None:
                raise ValueError("return_all is not supported if using sliding window.")
            # actual context depends on whether it's during sliding window or not
            # corner case at the beginning
            c = min(sliding_context_len, curr_token)
            # try biggest prediction chunk size
            h = min(length - curr_token, self.max_tokens - c)
            # chunk_size caps how many future tokens are diffused at once to save compute for causal model
            h = min(h, chunk_size) if chunk_size > 0 else h
            l = c + h
            pad = torch.zeros((batch_size, h, *x_shape))
            # context is last c tokens out of the sequence of generated/gt tokens
            # pad to length that's required by _sample_sequence
            context = torch.cat([xs_pred[:, -c:], pad.to(self.device)], 1)
            # calculate number of model generated tokens (not GT context tokens)
            generated_len = curr_token - max(curr_token - c, gt_len)
            # make context mask
            context_mask = torch.ones((batch_size, c), dtype=torch.long)
            if generated_len > 0:
                context_mask[:, -generated_len:] = 2
            pad = torch.zeros((batch_size, h), dtype=torch.long)
            context_mask = torch.cat([context_mask, pad.long()], 1).to(context.device)

            cond_len = l if self.use_causal_mask else self.max_tokens
            cond_slice = None
            if conditions is not None:
                cond_slice = conditions[:, curr_token - c : curr_token - c + cond_len]
            # import pdb; pdb.set_trace()   
            new_pred, record = self._sample_sequence(
                batch_size,
                length=l,
                context=context,
                context_mask=context_mask,
                conditions=cond_slice,
                guidance_fn=guidance_fn,
                reconstruction_guidance=reconstruction_guidance,
                history_guidance=history_guidance,
                return_all=return_all,
                pbar=pbar,
            )
            xs_pred = torch.cat([xs_pred, new_pred[:, -h:]], 1)
            curr_token = xs_pred.shape[1]
        pbar.close()
        return xs_pred, record

    def _extend_x_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Extend the tensor by adding dimensions at the end to match x_stacked_shape."""
        return rearrange(x, "... -> ..." + " 1" * len(self.x_shape))

    def _sample_sequence(
        self,
        batch_size: int,
        length: Optional[int] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        reconstruction_guidance: float = 0.0,
        history_guidance: Optional[HistoryGuidance] = None,
        return_all: bool = False,
        pbar: Optional[tqdm] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        The unified sampling method, with length up to maximum token size.
        context of length can be provided along with a mask to achieve conditioning.

        Args
        ----
        batch_size: int
            Batch size of the sampling process
        length: Optional[int]
            Number of frames in sampled sequence
            If None, fall back to length of context, and then fall back to `self.max_tokens`
        context: Optional[torch.Tensor], Shape (batch_size, length, *self.x_shape)
            Context tokens to condition on. Assumed to be same across batch.
            Tokens that are specified as context by `context_mask` will be used for conditioning,
            and the rest will be discarded.
        context_mask: Optional[torch.Tensor], Shape (batch_size, length)
            Mask for context
            0 = To be generated, 1 = Ground truth context, 2 = Generated context
            Some sampling logic may discriminate between ground truth and generated context.
        conditions: Optional[torch.Tensor], Shape (batch_size, length (causal) or self.max_tokens (noncausal), ...)
            Unprocessed external conditions for sampling
        guidance_fn: Optional[Callable]
            Guidance function for sampling
        history_guidance: Optional[HistoryGuidance]
            History guidance object that handles compositional generation
        return_all: bool
            Whether to return all steps of the sampling process
        Returns
        -------
        xs_pred: torch.Tensor, Shape (batch_size, length, *self.x_shape)
            Complete sequence containing context and generated tokens
        record: Optional[torch.Tensor], Shape (num_steps, batch_size, length, *self.x_shape)
            All recorded intermediate results during the sampling process
        """
        x_shape = self.x_shape

        if length is None:
            length = self.max_tokens if context is None else context.shape[1]
        if length > self.max_tokens:
            raise ValueError(
                f"length is expected to <={self.max_tokens}, got {length}."
            )

        if context is not None:
            if context_mask is None:
                raise ValueError("context_mask must be provided if context is given.")
            if context.shape[0] != batch_size:
                raise ValueError(
                    f"context batch size is expected to be {batch_size} but got {context.shape[0]}."
                )
            if context.shape[1] != length:
                raise ValueError(
                    f"context length is expected to be {length} but got {context.shape[1]}."
                )
            if tuple(context.shape[2:]) != tuple(x_shape):
                raise ValueError(
                    f"context shape not compatible with x_stacked_shape {x_shape}."
                )

        if context_mask is not None:
            if context is None:
                raise ValueError("context must be provided if context_mask is given. ")
            if context.shape[:2] != context_mask.shape:
                raise ValueError("context and context_mask must have the same shape.")

        if conditions is not None:
            if self.use_causal_mask and conditions.shape[1] != length:
                raise ValueError(
                    f"for causal models, conditions length is expected to be {length}, got {conditions.shape[1]}."
                )
            elif not self.use_causal_mask and conditions.shape[1] != self.max_tokens:
                raise ValueError(
                    f"for noncausal models, conditions length is expected to be {self.max_tokens}, got {conditions.shape[1]}."
                )

        horizon = length if self.use_causal_mask else self.max_tokens
        padding = horizon - length
        # import pdb; pdb.set_trace()
        # create initial xs_pred with noise
        xs_pred = torch.randn(
            (batch_size, horizon, *x_shape),
            device=self.device,
            generator=self.generator,
        )
        xs_pred = torch.clamp(xs_pred, -self.clip_noise, self.clip_noise)

        if context is None:
            # create empty context and zero context mask
            context = torch.zeros_like(xs_pred)
            context_mask = torch.zeros_like(
                (batch_size, horizon), dtype=torch.long, device=self.device
            )
        elif padding > 0:
            # pad context and context mask to reach horizon
            context_pad = torch.zeros(
                (batch_size, padding, *x_shape), device=self.device
            )
            # NOTE: In context mask, -1 = padding, 0 = to be generated, 1 = GT context, 2 = generated context
            context_mask_pad = -torch.ones(
                (batch_size, padding), dtype=torch.long, device=self.device
            )
            context = torch.cat([context, context_pad], 1)
            context_mask = torch.cat([context_mask, context_mask_pad], 1)

        if history_guidance is None:
            # by default, use conditional sampling
            history_guidance = HistoryGuidance.conditional(
                timesteps=self.timesteps,
            )

        # replace xs_pred's context frames with context
        xs_pred = torch.where(self._extend_x_dim(context_mask) >= 1, context, xs_pred)

        # generate scheduling matrix
        scheduling_matrix = self._generate_scheduling_matrix(
            horizon - padding,
            padding,
        )
        scheduling_matrix = scheduling_matrix.to(self.device)
        scheduling_matrix = repeat(scheduling_matrix, "m t -> m b t", b=batch_size)
        # fill context tokens' noise levels as -1 in scheduling matrix
        if not self.is_full_sequence:
            scheduling_matrix = torch.where(
                context_mask[None] >= 1, -1, scheduling_matrix
            )

        # prune scheduling matrix to remove identical adjacent rows
        diff = scheduling_matrix[1:] - scheduling_matrix[:-1]
        skip = torch.argmax((~reduce(diff == 0, "m b t -> m", torch.all)).float())
        scheduling_matrix = scheduling_matrix[skip:]

        record = [] if return_all else None

        if pbar is None:
            pbar = tqdm(
                total=scheduling_matrix.shape[0] - 1,
                initial=0,
                desc="Sampling with DFoT",
                leave=False,
            )
        # import pdb; pdb.set_trace()
        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels = scheduling_matrix[m]
            to_noise_levels = scheduling_matrix[m + 1]

            # update context mask by changing 0 -> 2 for fully generated tokens
            context_mask = torch.where(
                torch.logical_and(context_mask == 0, from_noise_levels == -1),
                2,
                context_mask,
            )

            # create a backup with all context tokens unmodified
            xs_pred_prev = xs_pred.clone()
            if return_all:
                record.append(xs_pred.clone())

            conditions_mask = None
            with history_guidance(context_mask) as history_guidance_manager:
                nfe = history_guidance_manager.nfe
                pbar.set_postfix(NFE=nfe)
                xs_pred, from_noise_levels, to_noise_levels, conditions_mask = (
                    history_guidance_manager.prepare(
                        xs_pred,
                        from_noise_levels,
                        to_noise_levels,
                        replacement_fn=self.diffusion_model.q_sample,
                        replacement_only=self.is_full_sequence,
                    )
                )

                if reconstruction_guidance > 0:

                    def composed_guidance_fn(
                        xk: torch.Tensor,
                        pred_x0: torch.Tensor,
                        alpha_cumprod: torch.Tensor,
                    ) -> torch.Tensor:
                        loss = (
                            F.mse_loss(pred_x0, context, reduction="none")
                            * alpha_cumprod.sqrt()
                        )
                        _context_mask = rearrange(
                            context_mask.bool(),
                            "b t -> b t" + " 1" * len(x_shape),
                        )
                        # scale inversely proportional to the number of context frames
                        loss = torch.sum(
                            loss
                            * _context_mask
                            / _context_mask.sum(dim=1, keepdim=True).clamp(min=1),
                        )
                        likelihood = -reconstruction_guidance * 0.5 * loss
                        return likelihood

                else:
                    composed_guidance_fn = guidance_fn

                # update xs_pred by DDIM or DDPM sampling
                xs_pred = self.diffusion_model.sample_step(
                    xs_pred,
                    from_noise_levels,
                    to_noise_levels,
                    self._process_conditions(
                        (
                            repeat(
                                conditions,
                                "b ... -> (b nfe) ...",
                                nfe=nfe,
                            ).clone()
                            if conditions is not None
                            else None
                        ),
                        from_noise_levels,
                    ),
                    conditions_mask,
                    guidance_fn=composed_guidance_fn,
                )

                xs_pred = history_guidance_manager.compose(xs_pred)

            # only replace the tokens being generated (revert context tokens)
            xs_pred = torch.where(
                self._extend_x_dim(context_mask) == 0, xs_pred, xs_pred_prev
            )
            pbar.update(1)

        if return_all:
            record.append(xs_pred.clone())
            record = torch.stack(record)
        if padding > 0:
            xs_pred = xs_pred[:, :-padding]
            record = record[:, :, :-padding] if return_all else None

        return xs_pred, record

    # ---------------------------------------------------------------------
    # Latent & Normalization Utils
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def _run_vae(
        self,
        x: Tensor,
        shape: str,
        vae_fn: Callable[[Tensor], Tensor],
    ) -> Tensor:
        """
        Helper function to run the VAE, either for encoding or decoding.
        - Requires shape to be a permutation of b, t, c, h, w.
        - Reshapes the input tensor to the required shape for the VAE, and reshapes the output back.
            - x: `shape` shape.
            - VideoVAE requires (b, c, t, h, w) shape.
            - ImageVAE requires (b, c, h, w) shape.
        - Split the input tensor into chunks of size cfg.vae.batch_size, to avoid memory errors.
        """
        x = rearrange(x, f"{shape} -> b c t h w")
        batch_size = x.shape[0]
        vae_batch_size = self.cfg.vae.batch_size
        # chunk the input tensor by vae_batch_size
        chunks = torch.chunk(x, (batch_size + vae_batch_size - 1) // vae_batch_size, 0)
        outputs = []
        for chunk in chunks:
            b = chunk.shape[0]
            if not self.is_latent_video_vae:
                chunk = rearrange(chunk, "b c t h w -> (b t) c h w")
            output = vae_fn(chunk)
            if not self.is_latent_video_vae:
                output = rearrange(output, "(b t) c h w -> b c t h w", b=b)
            outputs.append(output)
        return rearrange(torch.cat(outputs, 0), f"b c t h w -> {shape}")

    def _encode(self, x: Tensor, shape: str = "b t c h w") -> Tensor:
        return self._run_vae(
            x, shape, lambda y: self.vae.encode(2.0 * y - 1.0).sample()
        )

    def _decode(self, latents: Tensor, shape: str = "b t c h w") -> Tensor:
        return self._run_vae(
            latents,
            shape,
            lambda y: (
                self.vae.decode(y, self._n_tokens_to_n_frames(latents.shape[1]))
                if self.is_latent_video_vae
                else self.vae.decode(y)
            )
            * 0.5
            + 0.5,
        )

    def _normalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return (xs - mean) / std

    def _unnormalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return xs * std + mean

    # ---------------------------------------------------------------------
    # Checkpoint Utils
    # ---------------------------------------------------------------------

    def _uncompile_checkpoint(self, checkpoint: Dict[str, Any]):
        """Converts the state_dict if self.diffusion_model is compiled, to uncompiled."""
        if self.cfg.compile:
            checkpoint["state_dict"] = {
                k.replace("diffusion_model._orig_mod.", "diffusion_model."): v
                for k, v in checkpoint["state_dict"].items()
            }

    def _compile_checkpoint(self, checkpoint: Dict[str, Any]):
        """Converts the state_dict to the format expected by the compiled model."""
        if self.cfg.compile:
            checkpoint["state_dict"] = {
                k.replace("diffusion_model.", "diffusion_model._orig_mod."): v
                for k, v in checkpoint["state_dict"].items()
            }

    def _should_include_in_checkpoint(self, key: str) -> bool:
        return key.startswith("diffusion_model.model") or key.startswith(
            "diffusion_model._orig_mod.model" 
        ) or key.startswith(
            "diffusion_model._orig_mod.base_model"
        ) or key.startswith(
            "diffusion_model.base_model"
        ) or key.startswith(
                "diffusion_model._orig_mod.controlnet"
        ) or key.startswith("diffusion_model.controlnet")
        
            

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # 1. (Optionally) uncompile the model's state_dict before saving
        self._uncompile_checkpoint(checkpoint)
        # 2. Only save the meaningful keys defined by self._should_include_in_checkpoint
        # by default, only the model's state_dict is saved and metrics & registered buffes (e.g. diffusion schedule) are not discarded
        state_dict = checkpoint["state_dict"]
        for key in list(state_dict.keys()):
            if not self._should_include_in_checkpoint(key):
                del state_dict[key]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # 1. (Optionally) compile the model's state_dict before loading
        self._compile_checkpoint(checkpoint)
        # 2. (Optionally) swap the state_dict of the model with the EMA weights for inference
        # super().on_load_checkpoint(checkpoint)
        # 3. (Optionally) reset the optimizer states - for fresh finetuning or resuming training
        if self.cfg.checkpoint.reset_optimizer:
            checkpoint["optimizer_states"] = []

        # 4. Rewrite the state_dict of the checkpoint, only leaving meaningful keys
        # defined by self._should_include_in_checkpoint
        # also print out warnings when the checkpoint does not exactly match the expected format
        ## add for controlnet 
        # remap key 
        ckpt_with_controlnet_weights = False
        if self.cfg.compile:
            # model is compiled 
            base_str = "diffusion_model._orig_mod.model"
        else:
            base_str = "diffusion_model.model" 
        
        target_str = base_str+ ".base_model"
        if "controlnet" in self.cfg.backbone.name:
            for key in list(checkpoint["state_dict"].keys()):
                if  base_str in key:
                    # import pdb; pdb.set_trace()
                    new_key = key.replace(base_str, target_str)
                    checkpoint["state_dict"][new_key] = checkpoint["state_dict"][key]
                    # print(f"key {key} is remapped to {new_key}")
                    del checkpoint["state_dict"][key]
                if "control" in key:
                    ckpt_with_controlnet_weights = True
        # print("ckpt_with_controlnet_weights", ckpt_with_controlnet_weights)
        # import pdb; pdb.set_trace()
        new_state_dict = {}
        for key, value in self.state_dict().items():
            if (
                self._should_include_in_checkpoint(key)
                and key in checkpoint["state_dict"]
            ):
                new_state_dict[key] = checkpoint["state_dict"][key]
            else:
                new_state_dict[key] = value

        # print keys that are ignored from the checkpoint
        ignored_keys = [
            key
            for key in checkpoint["state_dict"].keys()
            if not self._should_include_in_checkpoint(key)
        ]
        if ignored_keys:
            rank_zero_print(
                cyan("The following keys are ignored from the checkpoint:"),
                ignored_keys,
            )
        # print keys that are not found in the checkpoint
        # import pdb; pdb.set_trace()
        missing_keys = [
            key
            for key in self.state_dict().keys()
            if self._should_include_in_checkpoint(key)
            and key not in checkpoint["state_dict"]
        ]
        
        if missing_keys:
            rank_zero_print(
                cyan("The following keys are not found in the checkpoint:"),
                missing_keys,
            )
            if self.cfg.checkpoint.strict:
                raise ValueError(
                    "Thus, the checkpoint cannot be loaded. To ignore this error, turn off strict checkpoint loading by setting `algorithm.checkpoint.strict=False`."
                )
            else:
                rank_zero_print(
                    cyan(
                        "Strict checkpoint loading is turned off, so using the initialized value for the missing keys."
                    )
                )
        # import pdb; pdb.set_trace()
        checkpoint["state_dict"] = new_state_dict

    def _load_ema_weights_to_state_dict(self, checkpoint: Dict[str, Any]) -> None:
        if (
            checkpoint.get("pretrained_ema", False)
            and len(checkpoint["optimizer_states"]) == 0
        ):
            # NOTE: for lightweight EMA-only ckpts for releasing pretrained models,
            # we already have EMA weights in the state_dict
            return
        ema_weights = checkpoint["optimizer_states"][0]["ema"]
        parameter_keys = [
            "diffusion_model." + k for k, _ in self.diffusion_model.named_parameters()
        ]
        assert len(parameter_keys) == len(
            ema_weights
        ), "Number of original weights and EMA weights do not match."
        for key, weight in zip(parameter_keys, ema_weights):
            checkpoint["state_dict"][key] = weight

    # ---------------------------------------------------------------------
    # Config Utils
    # ---------------------------------------------------------------------

    def check_video_vae_compatibility(self, cfg: DictConfig):
        """
        Check if the configuration is compatible with VideoVAE.
        Currently, it is not compatible with many functionalities, due to complicated shape/length changes.
        """
        assert (
            cfg.latent.type == "online"
        ), "Latents must be processed online when using VideoVAE."
        assert (
            cfg.external_cond_dim == 0
        ), "External conditions are not supported yet when using VideoVAE."
