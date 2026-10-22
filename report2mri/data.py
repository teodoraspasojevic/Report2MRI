"""Training data: (report, volume) pairs from a JSONL manifest of local NIfTI files.

Each line is one series:

    {"image": "path/to/series.nii.gz", "findings": "...", "impression": "...",
     "modality": "T1w", "plane": "AXIAL"}

`image` may be relative to the manifest. Each volume is reoriented to RAS, resampled trilinearly to
its (modality, plane) grid, intensity-normalised (0th-99.5th percentile of nonzero voxels to
[0, 1], unclipped) and centre-cropped/padded to the grid shape, with the anterior-posterior window
shifted posteriorly to compensate for defacing.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from .geometry import check_series, dhw_to_xyz, grid_dhw
from .text import conditioning_sections

REQUIRED_FIELDS = ("image", "modality", "plane")
POSTERIOR_SHIFT_MM = 15.0


def read_manifest(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = []
    with open(path) as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [f for f in REQUIRED_FIELDS if not row.get(f)]
            if missing:
                raise ValueError(f"{path}:{number} is missing {missing}")
            check_series(row["modality"], row["plane"])
            image = Path(row["image"])
            row["image"] = image if image.is_absolute() else path.parent / image
            if not row["image"].is_file():
                raise FileNotFoundError(f"{path}:{number}: image not found: {row['image']}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no series")
    return rows


def load_resampled(path, target_spacing_dhw) -> np.ndarray:
    """RAS-canonical volume resampled to `target_spacing_dhw`, as float32 (D, H, W) = (S, R, A)."""
    import nibabel as nib

    image = nib.as_closest_canonical(nib.load(str(path)))
    data = image.get_fdata().astype(np.float32)
    np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    zooms = image.header.get_zooms()
    spacing = (float(zooms[2]), float(zooms[0]), float(zooms[1]))
    tensor = torch.from_numpy(data.transpose(2, 0, 1)).unsqueeze(0).unsqueeze(0)
    shape = [int(tensor.shape[2 + i] * (spacing[i] / target_spacing_dhw[i])) for i in range(3)]
    return F.interpolate(tensor, size=shape, mode="trilinear", align_corners=False)[0, 0].numpy()


def normalize_percentile(data: np.ndarray, upper: float = 99.5) -> np.ndarray:
    nonzero = data != 0
    if nonzero.sum() > 0:
        low, high = np.percentile(data[nonzero], 0.0), np.percentile(data[nonzero], upper)
    else:
        low, high = data.min(), data.max()
    if high - low > 1e-8:
        return (data - low) / (high - low)
    return np.zeros_like(data)


def crop_or_pad(data: np.ndarray, target_shape, posterior_shift_voxels: int) -> np.ndarray:
    """Centre crop/zero-pad a (D, H, W) array; the W (anterior-posterior) window is shifted
    posteriorly by `posterior_shift_voxels`, clamped to the array."""
    tensor = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
    td, th, tw = target_shape
    d, h, w = tensor.shape
    d_start = max((d - td) // 2, 0)
    h_start = max((h - th) // 2, 0)
    w_start = min(max(w // 2 - posterior_shift_voxels - tw // 2, 0), max(w - tw, 0))
    tensor = tensor[d_start:d_start + td, h_start:h_start + th, w_start:w_start + tw]
    pads = []
    for axis in (2, 1, 0):
        missing = target_shape[axis] - tensor.shape[axis]
        pads.extend([missing // 2, missing - missing // 2])
    return F.pad(tensor, pads, value=0).numpy()


def preprocess(path, modality: str, plane: str, posterior_shift_mm: float = POSTERIOR_SHIFT_MM):
    """`(image [1, X, Y, Z] float32, spacing_mm_xyz)` on the (modality, plane) training grid."""
    shape, spacing = grid_dhw(modality, plane)
    shift = int(round(posterior_shift_mm / spacing[2]))
    data = crop_or_pad(normalize_percentile(load_resampled(path, spacing)), shape, shift)
    image = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)).unsqueeze(0)
    return image.permute(0, 2, 3, 1).contiguous(), tuple(float(v) for v in dhw_to_xyz(spacing))


class ReportVolumeDataset(Dataset):
    def __init__(self, manifest: str | Path, posterior_shift_mm: float = POSTERIOR_SHIFT_MM) -> None:
        self.rows = read_manifest(manifest)
        self.posterior_shift_mm = posterior_shift_mm

    def __len__(self) -> int:
        return len(self.rows)

    def bucket(self, index: int) -> tuple[str, str]:
        return self.rows[index]["modality"], self.rows[index]["plane"]

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image, spacing = preprocess(row["image"], row["modality"], row["plane"], self.posterior_shift_mm)
        return {
            "image": image,
            "sections": conditioning_sections(row.get("findings"), row.get("impression"),
                                              row["modality"], row["plane"], spacing),
            "modality": row["modality"],
            "spacing_mm": torch.tensor(spacing, dtype=torch.float32),
        }


def collate(batch: list[dict]) -> dict:
    shapes = {tuple(item["image"].shape) for item in batch}
    if len(shapes) != 1:
        raise ValueError(f"a batch mixes grid shapes {sorted(shapes)}; use BucketBatchSampler")
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "sections": [item["sections"] for item in batch],
        "modality": [item["modality"] for item in batch],
        "spacing_mm": torch.stack([item["spacing_mm"] for item in batch]),
    }


class BucketBatchSampler(Sampler):
    """Batches drawn from one (modality, plane) bucket at a time, since grid shapes differ.

    Each bucket is shuffled and split into batches (a remainder batch is dropped unless it is the
    bucket's only batch); every bucket's batches are then spread evenly over the epoch with a
    random phase, so consecutive batches carry different series types.
    """

    def __init__(self, dataset: ReportVolumeDataset, batch_size: int, seed: int = 0) -> None:
        self.batch_size = int(batch_size)
        self.seed = seed
        self.epoch = 0
        self.buckets: dict = {}
        for index in range(len(dataset)):
            self.buckets.setdefault(dataset.bucket(index), []).append(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(f"{self.seed}:{self.epoch}")
        per_bucket = {}
        for key, indices in self.buckets.items():
            indices = list(indices)
            rng.shuffle(indices)
            batches = [indices[s:s + self.batch_size] for s in range(0, len(indices), self.batch_size)]
            if len(batches) > 1 and len(batches[-1]) < self.batch_size:
                batches.pop()
            per_bucket[key] = batches
        total = sum(len(b) for b in per_bucket.values())
        schedule = []
        for batches in per_bucket.values():
            stride, phase = total / len(batches), rng.random()
            schedule.extend(((k + phase) * stride, batch) for k, batch in enumerate(batches))
        schedule.sort(key=lambda item: item[0])
        return iter([batch for _, batch in schedule])

    def __len__(self) -> int:
        n = 0
        for indices in self.buckets.values():
            full, remainder = divmod(len(indices), self.batch_size)
            n += full if full >= 1 else 1 if remainder else 0
        return n
