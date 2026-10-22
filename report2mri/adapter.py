"""Adapter checkpoints and weight resolution.

An adapter checkpoint holds only the trainable tensors (`adapter_state_dict`) plus what is needed to
check it against the frozen parts it was trained with: the adapter geometry, the text-encoder
identity, the base UNet's sha256 and the latent `scale_factor`. The base UNet, VAE and text encoders
are never stored in it.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch

ADAPTER_FORMAT = "mrrate_r2v_adapter_v1"

# The released adapter was trained against this exact base checkpoint.
BASE_UNET_REPO = "nvidia/NV-Generate-MR-Brain"
BASE_UNET_FILE = "models/diff_unet_3d_rflow-mr-brain_v0.pt"
BASE_UNET_SHA256 = "b49aa3162ff94b1852f81e520ed4165c5656000ba176c033063c35350b3305e9"
VAE_REPO = "nvidia/NV-Generate-CT"
VAE_FILE = "models/autoencoder_v1.pt"

ADAPTER_FILES = ("adapter.pt", "adapter.safetensors")
SLICE_TABLE_FILE = "native_spacing_table.json"
INPLANE_TABLE_FILE = "inplane_spacing_table.json"

# The encoder-identity fields that must agree between a checkpoint and the live encoders.
_IDENTITY_FIELDS = ("kind", "sections", "encoder_order", "encoder_dims", "pooling", "output_dim")


def sha256_file(path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class WeightPaths:
    adapter: Path | None              # None when only the frozen components are needed (training)
    base_unet: Path
    vae: Path
    text_encoders: tuple[Path, ...]   # one directory per encoder, in `text.ENCODERS` order
    slice_table: Path | None = None
    inplane_table: Path | None = None


def _require(path: Path, what: str, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}. {hint}")
    return path


def local_weight_paths(root: str | os.PathLike, require_adapter: bool = True) -> WeightPaths:
    """Paths inside a weights directory laid out as in the README."""
    from .text import ENCODERS

    root = Path(root)
    hint = "See 'Pretrained Checkpoint' in the README for the expected weights/ layout."
    _require(root, "weights directory", hint)
    adapter = None
    if require_adapter:
        adapter = next((root / name for name in ADAPTER_FILES if (root / name).is_file()), None)
        if adapter is None:
            raise FileNotFoundError(f"no {' or '.join(ADAPTER_FILES)} in {root}. {hint}")
    encoders = tuple(_require(root / spec.directory, f"text encoder {spec.hf_repo}", hint) for spec in ENCODERS)
    tables = [root / SLICE_TABLE_FILE, root / INPLANE_TABLE_FILE]
    return WeightPaths(
        adapter=adapter,
        base_unet=_require(root / Path(BASE_UNET_FILE).name, "base UNet", hint),
        vae=_require(root / Path(VAE_FILE).name, "VAE", hint),
        text_encoders=encoders,
        slice_table=tables[0] if tables[0].is_file() else None,
        inplane_table=tables[1] if tables[1].is_file() else None,
    )


def hub_weight_paths(repo_id: str, *, token=None, cache_dir=None) -> WeightPaths:
    """Download (or reuse from the Hugging Face cache) the adapter repo, the NVIDIA base model and
    VAE, and the three text encoders at their pinned revisions."""
    from huggingface_hub import hf_hub_download, snapshot_download

    from .text import ENCODERS

    adapter_dir = Path(snapshot_download(repo_id, allow_patterns=["adapter.pt", SLICE_TABLE_FILE, INPLANE_TABLE_FILE],
                                         token=token, cache_dir=cache_dir))
    base = Path(hf_hub_download(BASE_UNET_REPO, BASE_UNET_FILE, cache_dir=cache_dir))
    vae = Path(hf_hub_download(VAE_REPO, VAE_FILE, cache_dir=cache_dir))
    encoders = tuple(
        Path(snapshot_download(spec.hf_repo, revision=spec.revision, cache_dir=cache_dir,
                               allow_patterns=["*.json", "*.txt", spec.weights_file], ignore_patterns=["*/*"]))
        for spec in ENCODERS
    )
    return WeightPaths(adapter=adapter_dir / "adapter.pt", base_unet=base, vae=vae, text_encoders=encoders,
                       slice_table=adapter_dir / SLICE_TABLE_FILE, inplane_table=adapter_dir / INPLANE_TABLE_FILE)


def resolve_weights(source: str | os.PathLike, *, token=None, cache_dir=None) -> WeightPaths:
    """A local weights directory, or a Hugging Face model id such as `owner/Report2MRI`."""
    path = Path(source)
    if path.is_dir():
        return local_weight_paths(path)
    if re.fullmatch(r"[\w.-]+/[\w.-]+", str(source)) and not path.exists():
        return hub_weight_paths(str(source), token=token, cache_dir=cache_dir)
    raise FileNotFoundError(f"weights directory not found: {source} (expected a directory laid out as in the "
                            "README, or a Hugging Face model id)")


def load_adapter(path: str | os.PathLike) -> tuple[dict, dict]:
    """`(adapter_state_dict, metadata)` from an `adapter.pt` or `adapter.safetensors` file.

    A `.safetensors` file carries tensors only; its metadata is the released configuration.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"adapter checkpoint not found: {path}")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path)), {"base_checkpoint": {"sha256": BASE_UNET_SHA256}}
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != ADAPTER_FORMAT:
        found = payload.get("format") if isinstance(payload, dict) else type(payload).__name__
        raise RuntimeError(f"{path} is not a Report2MRI adapter checkpoint (format={found!r}, "
                           f"expected {ADAPTER_FORMAT!r})")
    if not isinstance(payload.get("adapter_state_dict"), dict):
        raise RuntimeError(f"{path} has no adapter_state_dict")
    metadata = {k: v for k, v in payload.items() if k != "adapter_state_dict"}
    return payload["adapter_state_dict"], metadata


def check_adapter_compatible(metadata: dict, encoder_identity: dict, base_sha256: str | None,
                             scale_factor: float | None, architecture: dict | None = None) -> None:
    """Refuse an adapter trained against a different base UNet, text-encoder setup, latent scale
    or adapter geometry. Fields the checkpoint does not record are not checked."""
    from .model import CONTEXT_DIM, CROSS_ATTENTION_DIM

    problems = []
    stored_sha = (metadata.get("base_checkpoint") or {}).get("sha256")
    if stored_sha and base_sha256 and stored_sha != base_sha256:
        problems.append(f"base UNet sha256 {base_sha256[:12]}... differs from the {stored_sha[:12]}... the "
                        f"adapter was trained on (use {BASE_UNET_FILE} from {BASE_UNET_REPO})")

    stored_identity = metadata.get("text_encoder") or {}
    for field in _IDENTITY_FIELDS:
        if field in stored_identity and stored_identity[field] != encoder_identity.get(field):
            problems.append(f"text encoder {field}: checkpoint {stored_identity[field]!r}, "
                            f"loaded {encoder_identity.get(field)!r}")
    for stored, live in zip(stored_identity.get("members") or [], encoder_identity.get("members") or []):
        for field in ("hf_repo", "max_length", "output_dim"):
            if field in stored and stored[field] != live.get(field):
                problems.append(f"text encoder {live.get('name')} {field}: checkpoint {stored[field]!r}, "
                                f"loaded {live.get(field)!r}")

    stored_scale = metadata.get("scale_factor")
    if stored_scale is not None and scale_factor is not None and float(stored_scale) != float(scale_factor):
        problems.append(f"scale_factor: checkpoint {stored_scale}, base UNet {scale_factor}")

    expected = architecture or {"context_dim": CONTEXT_DIM, "cross_attention_dim": CROSS_ATTENTION_DIM,
                                "conditioning_levels": [False, False, True, True], "condition_mid": True}
    config = metadata.get("config") or {}
    for field, value in expected.items():
        if field in config and config[field] != value:
            problems.append(f"adapter {field}: checkpoint {config[field]!r}, model {value!r}")
    if problems:
        raise RuntimeError("incompatible adapter checkpoint:\n  " + "\n  ".join(problems))


def save_adapter_checkpoint(path, model, *, text_identity: dict, base_sha256: str, scale_factor: float,
                            step: int, optimizer_step: int, epoch: int, loss: float | None,
                            training: dict, optimizer=None, lr_scheduler=None, rng_state=None) -> Path:
    """Write an adapter checkpoint. Only file names and hashes are recorded, never local paths."""
    from .model import adapter_parameter_names

    state = model.state_dict()
    payload = {
        "format": ADAPTER_FORMAT,
        "adapter_state_dict": {name: state[name].detach().cpu().clone() for name in adapter_parameter_names(model)},
        "config": {**model.architecture, "training": training},
        "text_encoder": text_identity,
        "base_checkpoint": {"path": Path(BASE_UNET_FILE).name, "sha256": base_sha256},
        "scale_factor": float(scale_factor),
        "step": int(step),
        "optimizer_step": int(optimizer_step),
        "epoch": int(epoch),
        "loss": loss,
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "lr_scheduler_state_dict": None if lr_scheduler is None else lr_scheduler.state_dict(),
        "rng_state": rng_state,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, str(tmp))
    os.replace(tmp, path)
    return path
