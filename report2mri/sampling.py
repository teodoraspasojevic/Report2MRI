"""Report -> volume inference.

Follows NV-Generate-CTMR's `scripts/diff_model_infer.py`: rectified-flow sampling of a latent at
`shape // 4`, the scheduler's own step, autocast on CUDA, NVIDIA's `ReconModel` under a Gaussian
sliding-window decoder, and the MR output range `[0, 1000]` as int16. The only change is
classifier-free guidance over the report as well as the modality:

    D = D_00 + s_mod * (D_m0 - D_00) + s_rep * (D_mr - D_m0)

where D_00 has neither modality nor report, D_m0 the modality only, and D_mr both. `s_rep = 0`
recovers NVIDIA's modality guidance exactly.
"""
from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism
from torch.amp import autocast

from .adapter import WeightPaths, check_adapter_compatible, load_adapter, resolve_weights, sha256_file
from .geometry import check_series, output_grid
from .model import build_unet, load_base_weights, read_base_checkpoint
from .nvidia import (LATENT_CHANNELS, LATENT_DIVISOR, MODALITY_CLASS_IDS, UNCONDITIONAL_CLASS_ID,
                     ReconModel, build_noise_scheduler, dynamic_infer, load_autoencoder)
from .output_calibration import SpacingTable, calibrate_grid
from .text import ReportEncoder, conditioning_sections, split_report

log = logging.getLogger("report2mri")

DEFAULT_STEPS = 30
DEFAULT_REPORT_GUIDANCE = 3.0
DEFAULT_MODALITY_GUIDANCE = 7.0
DEFAULT_SEED = 1234


@dataclass
class GenerationResult:
    volume: np.ndarray          # (X, Y, Z), int16 values in [0, 1000]
    spacing_mm: tuple           # (X, Y, Z)
    sections: dict              # the text the model was conditioned on


def guided_prediction(unet, x, timesteps, class_labels, spacing_tensor, context, context_mask,
                      modality_guidance: float, report_guidance: float) -> torch.Tensor:
    """One guided model output, with every guidance branch in a single batched UNet call."""
    batch = x.shape[0]
    labels = [class_labels]
    drops = [torch.ones(batch, dtype=torch.bool, device=x.device)]
    if modality_guidance != 0.0:
        labels.append(torch.full_like(class_labels, UNCONDITIONAL_CLASS_ID))
        drops.append(torch.ones(batch, dtype=torch.bool, device=x.device))
    if report_guidance != 0.0:
        labels.append(class_labels)
        drops.append(torch.zeros(batch, dtype=torch.bool, device=x.device))
    n = len(labels)
    outputs = unet(
        x=torch.cat([x] * n), timesteps=torch.cat([timesteps] * n),
        spacing_tensor=torch.cat([spacing_tensor] * n), context=torch.cat([context] * n),
        context_mask=torch.cat([context_mask] * n), class_labels=torch.cat(labels),
        context_drop_mask=torch.cat(drops),
    ).chunk(n)
    modality_only, rest = outputs[0], list(outputs[1:])
    guided = modality_only
    if modality_guidance != 0.0:
        unconditional = rest.pop(0)
        guided = unconditional + modality_guidance * (modality_only - unconditional)
    if report_guidance != 0.0:
        guided = guided + report_guidance * (rest.pop(0) - modality_only)
    return guided


def case_seed(seed: int, key: str) -> int:
    """A per-case seed, so that cases generated in one run start from different noise."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:4], "big")


def postprocess_mr(volume: np.ndarray) -> np.ndarray:
    """Decoder output in [0, 1] -> NVIDIA's MR range [0, 1000], clipped below at 0, as int16."""
    return np.int16(np.clip(volume * 1000.0, 0, None))


def save_nifti(volume: np.ndarray, spacing_mm: Sequence[float], path: str | Path) -> None:
    """Write `(X, Y, Z)` data with an axis-aligned affine whose diagonal is the spacing."""
    import nibabel as nib

    affine = np.eye(4)
    for i in range(3):
        affine[i, i] = float(spacing_mm[i])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume, affine=affine), str(path))


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Report2MRIPipeline:
    """Frozen text encoders + report-conditioned UNet + frozen VAE decoder."""

    def __init__(self, unet, autoencoder, text_encoder: ReportEncoder, scale_factor: float, device,
                 slice_table: SpacingTable | None = None, inplane_table: SpacingTable | None = None) -> None:
        self.device = torch.device(device)
        self.unet = unet.to(self.device).eval()
        self.autoencoder = autoencoder.to(self.device).eval() if autoencoder is not None else None
        self.text_encoder = text_encoder.to(self.device).eval()
        self.scale_factor = float(scale_factor)
        self.scheduler = build_noise_scheduler()
        self.slice_table = slice_table
        self.inplane_table = inplane_table

    @classmethod
    def from_pretrained(cls, weights: str | os.PathLike = "weights", device=None, *,
                        token: str | bool | None = None, cache_dir: str | None = None) -> "Report2MRIPipeline":
        """Load from a local weights directory (see README) or a Hugging Face model id."""
        device = torch.device(device) if device is not None else default_device()
        paths = resolve_weights(weights, token=token, cache_dir=cache_dir)
        return cls.from_paths(paths, device)

    @classmethod
    def from_paths(cls, paths: WeightPaths, device) -> "Report2MRIPipeline":
        device = torch.device(device)
        text_encoder = ReportEncoder.from_directories(paths.text_encoders)
        adapter_state, adapter_meta = load_adapter(paths.adapter)
        base_state, scale_factor = read_base_checkpoint(paths.base_unet)
        unet = build_unet(use_flash_attention=device.type == "cuda")
        check_adapter_compatible(adapter_meta, text_encoder.identity, sha256_file(paths.base_unet), scale_factor,
                                 unet.architecture)
        load_base_weights(unet, base_state, adapter_state)
        del base_state
        autoencoder = load_autoencoder(paths.vae, device=device)
        slice_table = SpacingTable.load(paths.slice_table, "thickness_mm") if paths.slice_table else None
        inplane_table = SpacingTable.load(paths.inplane_table, "inplane_mm") if paths.inplane_table else None
        return cls(unet, autoencoder, text_encoder, scale_factor, device, slice_table, inplane_table)

    @torch.no_grad()
    def sample_latent(self, sections: dict, modality: str, shape_xyz, spacing_mm_xyz, *,
                      seed: int | None = DEFAULT_SEED, steps: int = DEFAULT_STEPS,
                      report_guidance: float = DEFAULT_REPORT_GUIDANCE,
                      modality_guidance: float = DEFAULT_MODALITY_GUIDANCE) -> torch.Tensor:
        for axis, size in enumerate(shape_xyz):
            if size % (LATENT_DIVISOR * 8):
                raise ValueError(f"every output axis must be a multiple of 32, got {tuple(shape_xyz)}")
        if seed is not None:
            set_determinism(seed)
        latent = torch.randn((1, LATENT_CHANNELS, *(int(s) // LATENT_DIVISOR for s in shape_xyz)), device=self.device)
        self.scheduler.set_timesteps(num_inference_steps=steps,
                                     input_img_size_numel=int(torch.prod(torch.tensor(latent.shape[2:]))))
        timesteps = self.scheduler.timesteps
        next_timesteps = torch.cat((timesteps[1:], torch.tensor([0], dtype=timesteps.dtype)))

        context, context_mask = self.text_encoder.encode([sections])
        class_labels = torch.tensor([MODALITY_CLASS_IDS[modality]], dtype=torch.long, device=self.device)
        spacing = (torch.tensor(list(spacing_mm_xyz), dtype=torch.float32, device=self.device) * 1e2).unsqueeze(0)

        on_cuda = self.device.type == "cuda"
        with autocast("cuda" if on_cuda else "cpu", enabled=on_cuda):
            for t, next_t in zip(timesteps, next_timesteps):
                output = guided_prediction(
                    self.unet, latent, t.reshape(1).to(self.device, torch.float32), class_labels, spacing,
                    context, context_mask, modality_guidance, report_guidance)
                latent, _ = self.scheduler.step(output, t, latent, next_t)
        return latent.float()

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> np.ndarray:
        """Latent -> decoder output in ~[0, 1], `(X, Y, Z)` float32."""
        if self.autoencoder is None:
            raise RuntimeError("this pipeline has no autoencoder to decode with")
        recon = ReconModel(autoencoder=self.autoencoder, scale_factor=self.scale_factor).to(self.device)
        inferer = SlidingWindowInferer(roi_size=[80, 80, 80], sw_batch_size=1, progress=False, mode="gaussian",
                                       overlap=0.4, sw_device=self.device, device=self.device)
        on_cuda = self.device.type == "cuda"
        with autocast("cuda" if on_cuda else "cpu", enabled=on_cuda):
            volume = dynamic_infer(inferer, recon, latent)
        return volume.squeeze().float().cpu().numpy()

    def generate(self, report: str | None = None, *, modality: str, plane: str,
                 findings: str | None = None, impression: str | None = None,
                 seed: int | None = DEFAULT_SEED, steps: int = DEFAULT_STEPS,
                 report_guidance: float = DEFAULT_REPORT_GUIDANCE,
                 modality_guidance: float = DEFAULT_MODALITY_GUIDANCE,
                 calibrate: bool = True, case_key: str | None = None,
                 shape_xyz=None, spacing_mm_xyz=None) -> GenerationResult:
        """Generate one volume.

        Give either `report` (free text, split on `Findings:` / `Impression:` headings) or
        `findings` and/or `impression`. With `calibrate=True` the output is resampled onto an
        acquisition grid drawn from the spacing tables, keyed on `case_key`.
        """
        check_series(modality, plane)
        if report is not None and (findings is not None or impression is not None):
            raise ValueError("give either `report` or `findings`/`impression`, not both")
        if report is not None:
            parts = split_report(report)
            findings, impression = parts.get("findings"), parts.get("impression")
        if shape_xyz is None or spacing_mm_xyz is None:
            shape_xyz, spacing_mm_xyz = output_grid(modality, plane)
        sections = conditioning_sections(findings, impression, modality, plane, spacing_mm_xyz)

        latent = self.sample_latent(sections, modality, shape_xyz, spacing_mm_xyz, seed=seed, steps=steps,
                                    report_guidance=report_guidance, modality_guidance=modality_guidance)
        volume = postprocess_mr(self.decode(latent))
        spacing = tuple(float(v) for v in spacing_mm_xyz)
        if calibrate:
            if self.slice_table is None or self.inplane_table is None:
                raise RuntimeError("grid calibration needs native_spacing_table.json and inplane_spacing_table.json "
                                   "in the weights directory; disable it with calibrate=False / --no-grid-calibration")
            if not case_key:
                raise ValueError("grid calibration needs a case_key to draw the output grid from")
            volume, spacing = calibrate_grid(volume, spacing, modality, plane, case_key,
                                             self.slice_table, self.inplane_table)
        return GenerationResult(volume=volume, spacing_mm=spacing, sections=sections)
