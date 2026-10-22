"""The report-conditioned NV-Generate-MR-Brain diffusion UNet.

MONAI's `with_conditioning=True` would replace each attention level's `SpatialAttentionBlock` with a
`SpatialTransformer`, a module tree NVIDIA's pretrained weights no longer load onto. Instead the
pretrained tree is kept unchanged and cross-attention is added as new top-level modules:

    context_proj                report embedding (2560) -> cross_attention_dim (512)
    down/mid/up_cross_attn      zero-initialised cross-attention adapters, applied to the tensor
                                entering each conditioned block
    null_context                learned "no report" token for report dropout and guidance

Because every adapter's `proj_out` is zero-initialised, the untrained model reproduces the base
model exactly.
"""
from __future__ import annotations

import contextlib
import inspect
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi
from monai.networks.blocks import Convolution
from monai.networks.blocks.crossattention import CrossAttentionBlock
from monai.networks.nets.diffusion_model_unet import zero_module
from monai.utils.type_conversion import convert_to_tensor
from torch import nn

from .nvidia import UNET_CONFIG, allow_maisi_checkpoint_globals

CONTEXT_DIM = 2560
CROSS_ATTENTION_DIM = 512

# State-dict prefixes of everything this module adds to the base UNet.
ADAPTER_PREFIXES = ("context_proj.", "null_context", "down_cross_attn.", "mid_cross_attn.", "up_cross_attn.")

_MAISI_PARAMS = inspect.signature(DiffusionModelUNetMaisi.__init__).parameters


def sdpa_backend_guard(on_cuda: bool):
    """Exclude cuDNN's SDPA kernel, which returns non-finite gradients from a finite bf16/fp16
    forward at some latent shapes (e.g. 48^3). A no-op off CUDA."""
    if not on_cuda:
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])


class MaskedCrossAttention(CrossAttentionBlock):
    """MONAI's `CrossAttentionBlock` (same parameters and state-dict keys) with a key-padding mask.
    `context_mask` is True for tokens to attend to."""

    _BORROWED = ("to_q", "to_k", "to_v", "out_proj", "input_rearrange", "out_rearrange", "scale", "drop_output")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        missing = [name for name in self._BORROWED if not hasattr(self, name)]
        if missing:
            raise RuntimeError(f"MONAI's CrossAttentionBlock no longer provides {missing}")

    def forward(self, x, context=None, context_mask=None):
        kv = context if context is not None else x
        q = self.input_rearrange(self.to_q(x))
        k = self.input_rearrange(self.to_k(kv))
        v = self.input_rearrange(self.to_v(kv))
        if self.attention_dtype is not None:
            q, k, v = q.to(self.attention_dtype), k.to(self.attention_dtype), v.to(self.attention_dtype)
        attn_mask = None
        if context_mask is not None:
            attn_mask = context_mask.view(context_mask.shape[0], 1, 1, context_mask.shape[1])
        out = F.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=attn_mask, scale=self.scale,
            dropout_p=self.dropout_rate if self.training else 0.0,
        )
        out = self.out_rearrange(out.to(x.dtype))
        return self.drop_output(self.out_proj(out))


class ReportCrossAttentionAdapter(nn.Module):
    """`x + proj_out(attn(norm_attn(proj_in(norm(x)))))` with a zero-initialised `proj_out`."""

    def __init__(self, spatial_dims, in_channels, num_attention_heads, num_head_channels,
                 cross_attention_dim, norm_num_groups=32, norm_eps=1e-6, dropout=0.0,
                 upcast_attention=False) -> None:
        super().__init__()
        inner_dim = num_attention_heads * num_head_channels
        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.proj_in = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=inner_dim,
                                   strides=1, kernel_size=1, padding=0, conv_only=True)
        self.norm_attn = nn.LayerNorm(inner_dim)
        self.attn = MaskedCrossAttention(
            hidden_size=inner_dim, num_heads=num_attention_heads, hidden_input_size=inner_dim,
            context_input_size=cross_attention_dim, dim_head=num_head_channels, dropout_rate=dropout,
            attention_dtype=torch.float if upcast_attention else None,
        )
        self.proj_out = zero_module(Convolution(spatial_dims=spatial_dims, in_channels=inner_dim,
                                                out_channels=in_channels, strides=1, kernel_size=1,
                                                padding=0, conv_only=True))

    def forward(self, x, context, context_mask=None):
        h = self.proj_in(self.norm(x))
        batch, channels = h.shape[:2]
        spatial = h.shape[2:]
        h = h.reshape(batch, channels, -1).transpose(1, 2)
        h = self.attn(self.norm_attn(h), context=context, context_mask=context_mask)
        h = h.transpose(1, 2).reshape(batch, channels, *spatial)
        return x + self.proj_out(h)


class ContextProjection(nn.Module):
    """LayerNorm -> Linear -> GELU -> Linear -> LayerNorm, `(B, L, context_dim) -> (B, L, cross_attention_dim)`."""

    def __init__(self, context_dim: int, cross_attention_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else 4 * cross_attention_dim
        self.context_dim = context_dim
        self.net = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cross_attention_dim),
            nn.LayerNorm(cross_attention_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 3 or context.shape[-1] != self.context_dim:
            raise ValueError(f"context must be (B, L, {self.context_dim}), got {tuple(context.shape)}")
        return self.net(context)


class ReportConditionedUNet(DiffusionModelUNetMaisi):
    """`DiffusionModelUNetMaisi`, structurally unchanged, plus report cross-attention adapters.

    Adapters sit before every down block whose level has `conditioning_levels[i]` set, before the
    mirrored up blocks, and before the middle block. Defaults condition the base model's attention
    levels, which for NV-Generate-MR-Brain gives five adapters.
    """

    def __init__(self, context_dim: int = CONTEXT_DIM, cross_attention_dim: int = CROSS_ATTENTION_DIM,
                 conditioning_levels: Sequence[bool] | None = None, condition_mid: bool = True,
                 **maisi_kwargs) -> None:
        if maisi_kwargs.get("with_conditioning"):
            raise ValueError("with_conditioning=True would replace the pretrained attention blocks")
        super().__init__(**maisi_kwargs)

        def maisi_arg(name):
            return maisi_kwargs[name] if name in maisi_kwargs else _MAISI_PARAMS[name].default

        num_channels = list(self.block_out_channels)
        levels = list(self.attention_levels if conditioning_levels is None else conditioning_levels)
        if len(levels) != len(num_channels):
            raise ValueError(f"conditioning_levels has length {len(levels)}, expected {len(num_channels)}")
        self.conditioning_levels = tuple(bool(v) for v in levels)
        self.cross_attention_dim = cross_attention_dim

        def make_adapter(channels: int, level: int) -> ReportCrossAttentionAdapter:
            head_channels = self.num_head_channels[level]
            if head_channels <= 0 or channels % head_channels:
                raise ValueError(f"no attention heads at level {level} for {channels} channels")
            return ReportCrossAttentionAdapter(
                spatial_dims=maisi_arg("spatial_dims"), in_channels=channels,
                num_attention_heads=channels // head_channels, num_head_channels=head_channels,
                cross_attention_dim=cross_attention_dim, norm_num_groups=maisi_arg("norm_num_groups"),
                norm_eps=maisi_arg("norm_eps"), upcast_attention=maisi_arg("upcast_attention"),
            )

        # Channels of the tensor entering each block, keyed by level index on the way down and on
        # the way up.
        n_levels = len(num_channels)
        reversed_channels = list(reversed(num_channels))
        self.down_cross_attn = nn.ModuleDict({
            str(i): make_adapter(num_channels[max(i - 1, 0)], i)
            for i in range(n_levels) if self.conditioning_levels[i]
        })
        self.up_cross_attn = nn.ModuleDict({
            str(i): make_adapter(reversed_channels[max(i - 1, 0)], n_levels - 1 - i)
            for i in range(n_levels) if self.conditioning_levels[n_levels - 1 - i]
        })
        self.mid_cross_attn = make_adapter(num_channels[-1], n_levels - 1) if condition_mid else None
        self.context_proj = ContextProjection(context_dim, cross_attention_dim)
        self.null_context = nn.Parameter(torch.zeros(1, 1, cross_attention_dim))

    @property
    def architecture(self) -> dict:
        """The adapter geometry, as recorded in (and checked against) adapter checkpoints."""
        return {
            "context_dim": self.context_proj.context_dim,
            "cross_attention_dim": self.cross_attention_dim,
            "conditioning_levels": list(self.conditioning_levels),
            "condition_mid": self.mid_cross_attn is not None,
        }

    def prepare_context(self, batch_size, context, context_drop_mask=None, context_mask=None):
        """Project the report embedding. `context=None`, rows selected by `context_drop_mask` and
        rows with no unmasked token all become the learned null token."""
        if context is None:
            return self.null_context.expand(batch_size, 1, -1), None
        if context.shape[0] != batch_size:
            raise ValueError(f"context batch {context.shape[0]} != input batch {batch_size}")
        projected = self.context_proj(context)
        n_tokens = projected.shape[1]

        mask = None
        if context_mask is not None:
            if tuple(context_mask.shape) != (batch_size, n_tokens):
                raise ValueError(f"context_mask shape {tuple(context_mask.shape)} != {(batch_size, n_tokens)}")
            mask = context_mask.to(dtype=torch.bool, device=projected.device)

        drop = torch.zeros(batch_size, dtype=torch.bool, device=projected.device)
        if context_drop_mask is not None:
            if tuple(context_drop_mask.shape) != (batch_size,):
                raise ValueError(f"context_drop_mask shape {tuple(context_drop_mask.shape)} != {(batch_size,)}")
            drop = context_drop_mask.to(dtype=torch.bool, device=projected.device)
        if mask is not None:
            drop = drop | ~mask.any(dim=1)

        if bool(drop.any()):
            # A null token repeated L times attends identically to a single null token.
            null = self.null_context.expand_as(projected).to(projected.dtype)
            projected = torch.where(drop.view(-1, 1, 1), null, projected)
            if mask is not None:
                mask = mask | drop.view(-1, 1)
        return projected, mask

    def _condition(self, adapters, index, h, context, context_mask):
        key = str(index)
        return adapters[key](h, context=context, context_mask=context_mask) if key in adapters else h

    def forward(self, x, timesteps, context=None, class_labels=None, spacing_tensor=None,
                context_drop_mask=None, context_mask=None, **unused):
        """`context` is the raw `(B, L, context_dim)` report embedding (None = unconditional),
        `context_mask` a `(B, L)` bool (True = real token), `context_drop_mask` a `(B,)` bool
        selecting the null token per sample."""
        if any(value is not None for value in unused.values()):
            raise TypeError(f"unsupported inputs: {sorted(k for k, v in unused.items() if v is not None)}")
        with sdpa_backend_guard(x.is_cuda):
            context, context_mask = self.prepare_context(x.shape[0], context, context_drop_mask, context_mask)
            emb = self._get_time_and_class_embedding(x, timesteps, class_labels)
            emb = self._get_input_embeddings(emb, None, None, spacing_tensor)
            h = self.conv_in(x)

            res_samples = [h]
            for i, block in enumerate(self.down_blocks):
                h = self._condition(self.down_cross_attn, i, h, context, context_mask)
                h, block_res = block(hidden_states=h, temb=emb, context=None)
                res_samples.extend(block_res)

            if self.mid_cross_attn is not None:
                h = self.mid_cross_attn(h, context=context, context_mask=context_mask)
            h = self.middle_block(h, emb, None)

            for i, block in enumerate(self.up_blocks):
                idx = -len(block.resnets)
                block_res, res_samples = res_samples[idx:], res_samples[:idx]
                h = self._condition(self.up_cross_attn, i, h, context, context_mask)
                h = block(hidden_states=h, res_hidden_states_list=block_res, temb=emb, context=None)
            return convert_to_tensor(self.out(h))


def build_unet(use_flash_attention: bool = True) -> ReportConditionedUNet:
    """The released architecture: NV-Generate-MR-Brain geometry plus five adapters."""
    return ReportConditionedUNet(**dict(UNET_CONFIG, use_flash_attention=use_flash_attention))


def adapter_parameter_names(model: nn.Module) -> list[str]:
    return sorted(name for name, _ in model.named_parameters() if name.startswith(ADAPTER_PREFIXES))


def freeze_base(model: nn.Module) -> None:
    """Make exactly the adapter parameters trainable."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(ADAPTER_PREFIXES))


def read_base_checkpoint(path: str | Path) -> tuple[dict, float]:
    """`(unet_state_dict, scale_factor)` from NVIDIA's `diff_unet_3d_rflow-mr-brain_v0.pt`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"base UNet checkpoint not found: {path}")
    allow_maisi_checkpoint_globals()
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "unet_state_dict" not in payload or "scale_factor" not in payload:
        raise RuntimeError(f"{path} is not an NV-Generate diffusion UNet checkpoint "
                           "(expected 'unet_state_dict' and 'scale_factor')")
    return payload["unet_state_dict"], float(payload["scale_factor"])


def load_base_weights(model: ReportConditionedUNet, base_state_dict: dict,
                      adapter_state_dict: dict | None = None) -> None:
    """Load the frozen base weights and, optionally, trained adapter weights, strictly.

    The base checkpoint must supply every non-adapter tensor and nothing else; the adapter state
    (when given) must supply exactly the adapter tensors. Without one the adapters keep their
    initialisation.
    """
    adapter_names = set(adapter_parameter_names(model))
    overlap = sorted(adapter_names & set(base_state_dict))
    if overlap:
        raise RuntimeError(f"base checkpoint unexpectedly contains adapter tensors: {overlap[:5]}")
    if adapter_state_dict is None:
        current = model.state_dict()
        adapter_state_dict = {name: current[name] for name in adapter_names}
    elif set(adapter_state_dict) != adapter_names:
        missing = sorted(adapter_names - set(adapter_state_dict))
        unexpected = sorted(set(adapter_state_dict) - adapter_names)
        raise RuntimeError(f"adapter weights do not match the model: missing={missing[:5]} "
                           f"unexpected={unexpected[:5]}")
    model.load_state_dict({**base_state_dict, **adapter_state_dict}, strict=True)
