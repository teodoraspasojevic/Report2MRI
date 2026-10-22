"""Output-grid calibration: rewrite a generated volume onto an acquisition grid drawn from the
training split's statistics for its (modality, plane), keeping the physical field of view.

1. Slice axis: box-filter (slab-average) down to a slice thickness drawn from
   `native_spacing_table.json`. Never upsamples.
2. In-plane axes: cubic-spline upsample to a pixel spacing drawn from `inplane_spacing_table.json`.
   Never downsamples.

Draws are deterministic in a per-case key. Volumes are `(X, Y, Z) = (R, A, S)`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SLICE_AXIS_XYZ = {"AXIAL": 2, "CORONAL": 1, "SAGITTAL": 0}
INPLANE_AXES_XYZ = {"AXIAL": (0, 1), "SAGITTAL": (1, 2), "CORONAL": (0, 2)}
# The training split has no oblique series; an oblique acquisition is treated as a tilted axial.
PLANE_FALLBACK = {"OBLIQUE": "AXIAL"}
PLANE_CODE = {"AXIAL": "axi", "SAGITTAL": "sag", "CORONAL": "cor"}
# Buckets with fewer surveyed series fall back to the modality-pooled, then the global, bucket.
MIN_BUCKET_SERIES = 200
TOP_K = 8


class SpacingTable:
    """A surveyed spacing distribution per (modality, plane) bucket."""

    SCHEMAS = {"thickness_mm": "mrrate-native-spacing/2", "inplane_mm": "mrrate-inplane-spacing/1"}

    def __init__(self, payload: dict, field: str, top_k: int = TOP_K) -> None:
        if payload.get("schema") != self.SCHEMAS[field]:
            raise ValueError(f"spacing table schema {payload.get('schema')!r} != {self.SCHEMAS[field]!r}")
        self.buckets = payload["buckets"]
        self.field = field
        self.top_k = max(1, int(top_k))

    @classmethod
    def load(cls, path: str | Path, field: str) -> "SpacingTable":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"spacing table not found: {path}")
        return cls(json.loads(path.read_text()), field)

    def resolve(self, modality: str, plane: str) -> dict:
        plane = PLANE_FALLBACK.get(plane, plane)
        chain = [f"{modality}|{PLANE_CODE[plane]}", f"{modality}|__pooled__", "__pooled__|__pooled__"]
        for key in chain:
            info = self.buckets.get(key)
            if info and info["n_series"] >= MIN_BUCKET_SERIES and info[self.field]:
                return info
        raise KeyError(f"no usable spacing bucket for ({modality!r}, {plane!r}); tried {chain}")

    def draw(self, modality: str, plane: str, key: str) -> float:
        """Sample a spacing from the bucket's top-k entries, renormalised, keyed on `key`."""
        entries = self.resolve(modality, plane)[self.field][: self.top_k]
        u = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") / float(1 << 64)
        total = sum(e["share"] for e in entries)
        cumulative = 0.0
        for entry in entries:
            cumulative += entry["share"] / total
            if u < cumulative:
                return float(entry["value"])
        return float(entries[-1]["value"])


def _slab_average(volume: np.ndarray, axis: int, n_out: int) -> np.ndarray:
    """Exact box-filter average of one axis from n_in to n_out samples (any ratio)."""
    n_in = volume.shape[axis]
    if n_out == n_in:
        return volume
    moved = np.moveaxis(volume, axis, 0).astype(np.float64, copy=False)
    edges = np.linspace(0.0, float(n_in), n_out + 1)
    cumulative = np.concatenate([np.zeros((1,) + moved.shape[1:]), np.cumsum(moved, axis=0)], axis=0)
    lo = np.clip(np.floor(edges).astype(np.int64), 0, n_in - 1)
    frac = (edges - lo).reshape((-1,) + (1,) * (moved.ndim - 1))
    integral = cumulative[lo] * (1.0 - frac) + cumulative[lo + 1] * frac
    widths = np.diff(edges).reshape((-1,) + (1,) * (moved.ndim - 1))
    out = (integral[1:] - integral[:-1]) / widths
    return np.moveaxis(out, 0, axis).astype(volume.dtype, copy=False)


def to_slice_thickness(volume: np.ndarray, spacing_xyz, plane: str, thickness_mm: float):
    """Slab-average the slice axis to ~`thickness_mm` with the FOV preserved. A target finer than
    the current grid leaves the volume unchanged."""
    axis = SLICE_AXIS_XYZ[PLANE_FALLBACK.get(plane, plane)]
    spacing = [float(v) for v in spacing_xyz]
    fov_mm = volume.shape[axis] * spacing[axis]
    n = max(1, int(round(fov_mm / thickness_mm)))
    if n >= volume.shape[axis]:
        return volume, tuple(spacing)
    spacing[axis] = fov_mm / n
    return _slab_average(volume, axis, n), tuple(spacing)


def to_inplane_spacing(volume: np.ndarray, spacing_xyz, plane: str, target_mm: float):
    """Cubic-spline upsample both in-plane axes to ~`target_mm` with the FOV preserved. A target
    coarser than the current grid leaves that axis unchanged."""
    from scipy.ndimage import zoom

    axes = INPLANE_AXES_XYZ[PLANE_FALLBACK.get(plane, plane)]
    shape = list(volume.shape)
    spacing = [float(v) for v in spacing_xyz]
    for axis in axes:
        fov_mm = shape[axis] * spacing[axis]
        n = max(1, int(round(fov_mm / target_mm)))
        if n > shape[axis]:
            shape[axis], spacing[axis] = n, fov_mm / n
    if tuple(shape) == volume.shape:
        return volume, tuple(spacing)
    factors = [shape[a] / volume.shape[a] if a in axes else 1.0 for a in range(volume.ndim)]
    out = zoom(volume.astype(np.float64, copy=False), factors, order=3, mode="nearest")
    if out.shape != tuple(shape):
        # `zoom` can round one voxel off the planned shape; crop or zero-pad back to the plan.
        fitted = np.zeros(shape, dtype=out.dtype)
        region = tuple(slice(0, min(s, t)) for s, t in zip(out.shape, shape))
        fitted[region] = out[region]
        out = fitted
    return out.astype(volume.dtype, copy=False), tuple(spacing)


def calibrate_grid(volume: np.ndarray, spacing_xyz, modality: str, plane: str, case_key: str,
                   slice_table: SpacingTable, inplane_table: SpacingTable):
    """Both rewrites, slice axis first. Returns `(volume, spacing_xyz)`."""
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D (X, Y, Z) volume, got shape {volume.shape}")
    thickness = slice_table.draw(modality, plane, case_key)
    volume, spacing = to_slice_thickness(volume, spacing_xyz, plane, thickness)
    inplane = inplane_table.draw(modality, plane, f"inplane:{case_key}")
    return to_inplane_spacing(volume, spacing, plane, inplane)
