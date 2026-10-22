"""The voxel grid each (modality, plane) is generated and trained on.

Shapes are NVIDIA's recommended NV-Generate-MR-Brain field of view per (modality, plane), rounded
to the nearest multiple of 32 (every UNet axis must be one); the spacing is then FOV / shape, so the
physical FOV matches NVIDIA's exactly. Everything public is in (X, Y, Z) = (R, A, S) order.
"""
from __future__ import annotations

MODALITIES = ("T1w", "T2w", "FLAIR", "SWI")
PLANES = ("AXIAL", "SAGITTAL", "CORONAL", "OBLIQUE")

UNET_SPATIAL_MULTIPLE = 32

# NV-Generate-CTMR docs/inference.md "Recommended FOV", in (D, H, W) = (S, R, A) mm.
_FOV_MM_DHW = {
    ("T1w", "AXIAL"): (174.0, 240.0, 240.0),
    ("T1w", "SAGITTAL"): (250.0, 176.0, 250.0),
    ("T1w", "CORONAL"): (240.0, 240.0, 200.0),
    ("T2w", "AXIAL"): (158.0, 240.0, 240.0),
    ("T2w", "SAGITTAL"): (240.0, 162.0, 240.0),
    ("T2w", "CORONAL"): (200.0, 200.0, 180.0),
    ("FLAIR", "AXIAL"): (175.0, 250.0, 250.0),
    ("FLAIR", "SAGITTAL"): (250.0, 176.0, 250.0),
    ("FLAIR", "CORONAL"): (250.0, 250.0, 200.0),
    ("SWI", "AXIAL"): (145.0, 230.0, 230.0),
    ("SWI", "SAGITTAL"): (230.0, 140.0, 230.0),
    ("SWI", "CORONAL"): (230.0, 230.0, 155.0),
}
# Any other combination (e.g. OBLIQUE): NV-Generate-MR-Brain's default 256^3 at 1 mm.
_FALLBACK_FOV_MM = (256.0, 256.0, 256.0)


def dhw_to_xyz(values):
    return (values[1], values[2], values[0])


def grid_dhw(modality: str, plane: str) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """`(shape, spacing_mm)` in internal (D, H, W) order."""
    fov = _FOV_MM_DHW.get((modality, plane), _FALLBACK_FOV_MM)
    shape = tuple(max(UNET_SPATIAL_MULTIPLE, int(round(f / UNET_SPATIAL_MULTIPLE)) * UNET_SPATIAL_MULTIPLE)
                  for f in fov)
    spacing = tuple(f / s for f, s in zip(fov, shape))
    return shape, spacing


def output_grid(modality: str, plane: str) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """`(shape_xyz, spacing_mm_xyz)` the model generates for this (modality, plane)."""
    check_series(modality, plane)
    shape, spacing = grid_dhw(modality, plane)
    return tuple(int(v) for v in dhw_to_xyz(shape)), tuple(float(v) for v in dhw_to_xyz(spacing))


def check_series(modality: str, plane: str) -> None:
    if modality not in MODALITIES:
        raise ValueError(f"unsupported modality {modality!r}; choose from {MODALITIES}")
    if plane not in PLANES:
        raise ValueError(f"unsupported plane {plane!r}; choose from {PLANES}")
