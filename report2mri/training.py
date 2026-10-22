"""Adapter training with NVIDIA's own objective (NV-Generate-CTMR `scripts/diff_model_train.py`):
an L1 loss on the rectified-flow velocity `x0 - noise`, Adam and a power-2 `PolynomialLR`.

Only the adapters and the context projection are trained; the UNet, VAE and text encoders stay
frozen. Latents are encoded on the fly by the frozen VAE and scaled by the base checkpoint's
`scale_factor`. Modality labels are dropped with NVIDIA's `augment_modality_label` and reports are
replaced by the learned null token with probability `report_dropout`, both for classifier-free
guidance.
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.amp import autocast

from .adapter import load_adapter, save_adapter_checkpoint
from .model import adapter_parameter_names, freeze_base
from .nvidia import MODALITY_CLASS_IDS, augment_modality_label

log = logging.getLogger("report2mri.training")


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    epochs: int = 2
    max_steps: int | None = None          # micro-steps for this invocation (smoke runs)
    batch_size: int = 8
    grad_accumulation_steps: int = 4
    grad_clip_norm: float | None = 1.0
    amp: bool = True                      # bfloat16 autocast on CUDA
    report_dropout: float = 0.1
    modality_dropout: float = 0.1
    seed: int = 0
    log_every: int = 10
    save_every_steps: int | None = None   # optimizer steps between periodic checkpoints
    max_consecutive_nonfinite: int = 5

    def __post_init__(self) -> None:
        if self.grad_accumulation_steps < 1:
            raise ValueError("grad_accumulation_steps must be >= 1")
        for name in ("report_dropout", "modality_dropout"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


class AdapterTrainer:
    def __init__(self, unet, text_encoder, autoencoder, noise_scheduler, scale_factor: float, base_sha256: str,
                 config: TrainingConfig, device, output_dir, rank: int = 0) -> None:
        self.unet = unet
        self.text_encoder = text_encoder
        self.autoencoder = autoencoder
        self.noise_scheduler = noise_scheduler
        self.scale_factor = float(scale_factor)
        self.base_sha256 = base_sha256
        self.config = config
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.rank = rank

        torch.set_float32_matmul_precision("highest")
        torch.manual_seed(config.seed)
        freeze_base(self.model)
        if autoencoder is not None:
            autoencoder.requires_grad_(False).eval()
        trainable = [p for p in self.unet.parameters() if p.requires_grad]
        expected = {id(p) for n, p in self.model.named_parameters() if n in set(adapter_parameter_names(self.model))}
        if {id(p) for p in trainable} != expected or any(p.requires_grad for p in text_encoder.parameters()):
            raise RuntimeError("only the adapter parameters may be trainable")
        self.optimizer = torch.optim.Adam(trainable, lr=config.lr)
        self.lr_scheduler = None
        self.dropout_generator = torch.Generator().manual_seed(config.seed + 12345)
        self.step = 0              # micro-steps
        self.optimizer_step = 0
        self.epoch = 0
        self.start_epoch = 0
        self.skipped_steps = 0
        self._nonfinite_streak = 0
        n_trainable = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in self.model.parameters())
        log.info("trainable parameters: %d of %d (%.2f%%)", n_trainable, n_total, 100 * n_trainable / n_total)

    @property
    def model(self):
        return getattr(self.unet, "module", self.unet)

    @property
    def amp_enabled(self) -> bool:
        return self.config.amp and self.device.type == "cuda"

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if any(s % 32 for s in images.shape[2:]):
            raise ValueError(f"image grid {tuple(images.shape[2:])} is not a multiple of 32")
        with autocast("cuda" if images.is_cuda else "cpu", enabled=self.amp_enabled, dtype=torch.bfloat16):
            latent = self.autoencoder.encode_stage_2_inputs(images.float())
        return latent.float() * self.scale_factor

    def prepare_batch(self, batch: dict) -> dict:
        """`batch["image"]` is a volume batch, or already a latent batch when there is no VAE."""
        images = batch["image"].to(self.device)
        latents = self.encode_images(images) if self.autoencoder is not None else images.float()
        spacing = batch["spacing_mm"].to(self.device).float() * 1e2
        labels = torch.tensor([MODALITY_CLASS_IDS[m] for m in batch["modality"]], dtype=torch.long, device=self.device)
        labels = augment_modality_label(labels, prob=self.config.modality_dropout)
        context, context_mask = self.text_encoder.encode(batch["sections"])
        drop = (torch.rand(latents.shape[0], generator=self.dropout_generator) < self.config.report_dropout)
        return {"latents": latents, "spacing_tensor": spacing, "class_labels": labels,
                "context": context.to(self.device), "context_mask": context_mask.to(self.device),
                "context_drop_mask": drop.to(self.device)}

    def train_step(self, batch: dict) -> dict:
        self.unet.train()
        inputs = self.prepare_batch(batch)
        latents = inputs.pop("latents")
        with autocast("cuda" if self.device.type == "cuda" else "cpu", enabled=self.amp_enabled, dtype=torch.bfloat16):
            noise = torch.randn_like(latents)
            timesteps = self.noise_scheduler.sample_timesteps(latents)
            noisy = self.noise_scheduler.add_noise(original_samples=latents, noise=noise, timesteps=timesteps)
            output = self.unet(x=noisy, timesteps=timesteps, **inputs)
            loss = torch.nn.functional.l1_loss(output.float(), (latents - noise).float())

        if not torch.isfinite(loss):
            self._nonfinite_streak += 1
            log.error("non-finite loss at optimizer step %d (%d in a row)", self.optimizer_step, self._nonfinite_streak)
            if self._nonfinite_streak >= self.config.max_consecutive_nonfinite:
                raise RuntimeError(f"loss was non-finite for {self._nonfinite_streak} consecutive steps")
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            return {"loss": float("nan"), "stepped": False, "grad_norm": None}
        self._nonfinite_streak = 0

        (loss / self.config.grad_accumulation_steps).backward()
        stepped, grad_norm = False, None
        if (self.step + 1) % self.config.grad_accumulation_steps == 0:
            trainable = [p for p in self.unet.parameters() if p.requires_grad]
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, self.config.grad_clip_norm or float("inf")))
            if math.isfinite(grad_norm):
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer_step += 1
                stepped = True
            else:
                self.skipped_steps += 1
                log.warning("skipped optimizer step: gradient norm is %s", grad_norm)
            self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        return {"loss": float(loss.detach()), "stepped": stepped, "grad_norm": grad_norm,
                "lr": self.optimizer.param_groups[0]["lr"]}

    def fit(self, loader) -> dict:
        start_step = self.step
        total_micro = self.config.max_steps or self.config.epochs * len(loader)
        if total_micro < self.config.grad_accumulation_steps:
            raise ValueError(f"{total_micro} batches per rank in total is fewer than --grad-accumulation-steps "
                             f"{self.config.grad_accumulation_steps}, so no optimizer step would be taken")
        total = total_micro // self.config.grad_accumulation_steps
        if self.lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=total, power=2.0)
        elif self.lr_scheduler.total_iters != total:
            # Otherwise the restored schedule would reach 0 early and the rest of the run would train nothing.
            raise RuntimeError(
                f"the resumed learning-rate schedule spans {self.lr_scheduler.total_iters} optimizer steps, but this "
                f"run is configured for {total}; resume with the original --epochs/--max-steps, --batch-size, "
                "--grad-accumulation-steps and number of GPUs")
        losses = []
        for epoch in range(self.start_epoch, self.config.epochs):
            self.epoch = epoch
            for component in (getattr(loader, "batch_sampler", None), getattr(loader, "sampler", None)):
                if hasattr(component, "set_epoch"):
                    component.set_epoch(epoch)
            for batch in loader:
                metrics = self.train_step(batch)
                losses.append(metrics["loss"])
                if metrics["stepped"]:
                    if self.rank == 0 and self.optimizer_step % max(self.config.log_every, 1) == 0:
                        log.info("epoch %d step %d loss %.5f lr %.3e grad_norm %.4f", epoch + 1, self.optimizer_step,
                                 metrics["loss"], metrics["lr"], metrics["grad_norm"])
                    every = self.config.save_every_steps
                    if every and self.optimizer_step % every == 0:
                        self.save(self.output_dir / f"adapter_step{self.optimizer_step:07d}.pt", metrics["loss"])
                if self.config.max_steps and self.step - start_step >= self.config.max_steps:
                    break
            if self.config.max_steps and self.step - start_step >= self.config.max_steps:
                break
        self.save(self.output_dir / "adapter_last.pt", losses[-1] if losses else None)
        return {"micro_steps": self.step, "optimizer_steps": self.optimizer_step, "epochs": self.epoch + 1,
                "skipped_steps": self.skipped_steps, "final_loss": losses[-1] if losses else None}

    def save(self, path, loss) -> None:
        if self.rank != 0:
            return
        save_adapter_checkpoint(
            path, self.model, text_identity=self.text_encoder.identity, base_sha256=self.base_sha256,
            scale_factor=self.scale_factor, step=self.step, optimizer_step=self.optimizer_step, epoch=self.epoch,
            loss=loss, training=asdict(self.config), optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
            rng_state={"torch": torch.get_rng_state(), "dropout_generator": self.dropout_generator.get_state(),
                       "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        )
        log.info("saved %s", path)

    def resume(self, path) -> None:
        """Continue from a checkpoint written by `save`; training restarts at the next epoch."""
        from .adapter import check_adapter_compatible

        state, meta = load_adapter(path)
        check_adapter_compatible(meta, self.text_encoder.identity, self.base_sha256, self.scale_factor,
                                 self.model.architecture)
        names = set(adapter_parameter_names(self.model))
        if set(state) != names:
            raise RuntimeError(f"{path} does not hold this model's adapter tensors")
        self.model.load_state_dict({**self.model.state_dict(), **state}, strict=True)
        if meta.get("optimizer_state_dict") is None or meta.get("lr_scheduler_state_dict") is None:
            raise RuntimeError(f"{path} has no optimizer/scheduler state to resume from")
        self.optimizer.load_state_dict(meta["optimizer_state_dict"])
        self.lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=1, power=2.0)
        self.lr_scheduler.load_state_dict(meta["lr_scheduler_state_dict"])
        rng = meta.get("rng_state") or {}
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if rng.get("dropout_generator") is not None:
            self.dropout_generator.set_state(rng["dropout_generator"])
        if rng.get("cuda") is not None and torch.cuda.is_available() and len(rng["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(rng["cuda"])
        self.step, self.optimizer_step = int(meta["step"]), int(meta["optimizer_step"])
        self.epoch = int(meta["epoch"])
        self.start_epoch = self.epoch + 1
        log.info("resumed from %s at optimizer step %d; continuing with epoch %d", path, self.optimizer_step,
                 self.start_epoch + 1)
