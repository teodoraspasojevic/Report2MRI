"""Frozen NVIDIA components: NV-Generate-MR-Brain diffusion UNet geometry, the MAISI VAE and the
rectified-flow scheduler.

The configuration values and the functions marked below are taken from NV-Generate-CTMR
(https://github.com/nvidia-medtech/NV-Generate-CTMR, `configs/config_network_rflow.json`,
`configs/modality_mapping.json`, `scripts/utils.py`, `scripts/utils_infer.py`,
`scripts/diff_model_train.py`), Copyright (c) MONAI Consortium, Apache License 2.0.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.networks.schedulers.rectified_flow import RFlowScheduler

# `diffusion_unet_def` of config_network_rflow.json.
UNET_CONFIG = dict(
    spatial_dims=3,
    in_channels=4,
    out_channels=4,
    num_channels=[64, 128, 256, 512],
    attention_levels=[False, False, True, True],
    num_head_channels=[0, 0, 32, 32],
    num_res_blocks=2,
    use_flash_attention=True,
    include_top_region_index_input=False,
    include_bottom_region_index_input=False,
    include_spacing_input=True,
    num_class_embeds=128,
    resblock_updown=True,
    include_fc=True,
)

# `autoencoder_def` of config_network_rflow.json.
VAE_CONFIG = dict(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    latent_channels=4,
    num_channels=[64, 128, 256],
    num_res_blocks=[2, 2, 2],
    norm_num_groups=32,
    norm_eps=1e-06,
    attention_levels=[False, False, False],
    with_encoder_nonlocal_attn=False,
    with_decoder_nonlocal_attn=False,
    use_checkpointing=False,
    use_convtranspose=False,
    norm_float16=True,
    num_splits=4,
    dim_split=1,
)

# `noise_scheduler` of config_network_rflow.json.
SCHEDULER_CONFIG = dict(
    num_train_timesteps=1000,
    use_discrete_timesteps=False,
    use_timestep_transform=True,
    sample_method="uniform",
    scale=1.4,
)

LATENT_CHANNELS = 4
# Output voxels per latent voxel along each axis: 2 ** (len(num_channels) - 2).
LATENT_DIVISOR = 4

# Class ids from modality_mapping.json. MR-RATE volumes are defaced but not skull-stripped, so the
# `*_skull_stripped` ids are not used.
MODALITY_CLASS_IDS = {"T1w": 9, "T2w": 10, "FLAIR": 11, "SWI": 20}
UNCONDITIONAL_CLASS_ID = 0


def build_noise_scheduler() -> RFlowScheduler:
    return RFlowScheduler(**SCHEDULER_CONFIG)


def allow_maisi_checkpoint_globals() -> None:
    """NVIDIA's checkpoints pickle `scale_factor` as a MONAI `MetaTensor`; allow-list it so the
    files load with `weights_only=True`."""
    from monai.data.meta_tensor import MetaTensor
    from monai.utils.enums import TraceKeys

    torch.serialization.add_safe_globals([MetaTensor, TraceKeys])


def load_autoencoder(path: str | Path, device="cpu") -> AutoencoderKlMaisi:
    """The frozen MAISI VAE (`autoencoder_v1.pt`), loaded strictly.

    Its half-precision group norms rely on CUDA autocast, so off CUDA they run in float32.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {path}")
    allow_maisi_checkpoint_globals()
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "unet_state_dict" in state:
        state = state["unet_state_dict"]
    autoencoder = AutoencoderKlMaisi(**dict(VAE_CONFIG, norm_float16=torch.device(device).type == "cuda"))
    autoencoder.load_state_dict(state, strict=True)
    autoencoder.requires_grad_(False)
    return autoencoder.to(device).eval()


# -- verbatim from NV-Generate-CTMR ------------------------------------------------------------


def dynamic_infer(inferer, model, images):
    """Perform dynamic inference using a model and an inferer, typically a monai SlidingWindowInferer.

    Determines whether to use the model directly or the provided inferer, based on input size.
    """
    if torch.numel(images[0:1, 0:1, ...]) <= math.prod(inferer.roi_size):
        return model(images)
    else:
        spatial_dims = images.shape[2:]
        orig_roi = inferer.roi_size
        if len(orig_roi) != len(spatial_dims):
            raise ValueError(f"ROI length ({len(orig_roi)}) does not match spatial dimensions ({len(spatial_dims)}).")
        adjusted_roi = [min(roi_dim, img_dim) for roi_dim, img_dim in zip(orig_roi, spatial_dims)]
        inferer.roi_size = adjusted_roi
        output = inferer(network=model, inputs=images)
        inferer.roi_size = orig_roi
        return output


class ReconModel(torch.nn.Module):
    """Reconstructs images from latent representations via the (frozen) autoencoder."""

    def __init__(self, autoencoder, scale_factor):
        super().__init__()
        self.autoencoder = autoencoder
        self.scale_factor = scale_factor

    def forward(self, z):
        return self.autoencoder.decode_stage_2_outputs(z / self.scale_factor)


def augment_modality_label(modality_tensor: torch.Tensor, prob: float = 0.1) -> torch.Tensor:
    """Augments the modality tensor by randomly modifying certain elements based on `prob`."""
    mask_ct = (modality_tensor < 8) & (modality_tensor >= 2)
    prob_ct = torch.rand(modality_tensor.size(), device=modality_tensor.device) < prob
    modality_tensor[mask_ct & prob_ct] = 1

    mask_mri = modality_tensor >= 9
    prob_mri = torch.rand(modality_tensor.size(), device=modality_tensor.device) < prob
    modality_tensor[mask_mri & prob_mri] = 8

    mask_zero = torch.rand(modality_tensor.size(), device=modality_tensor.device) > prob
    modality_tensor = modality_tensor * mask_zero.long()

    return modality_tensor
