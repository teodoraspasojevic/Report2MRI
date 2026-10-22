#!/usr/bin/env python3
"""VLM3D MRI report-to-volume submission: /input/prompts.json -> /output/<input_image_name>.nii.gz.

Each prompt is `{"input_image_name": "<study>_<modality>-raw-<plane>[-N]", "report": "..."}`. The
modality and plane come from the name, the report is used as given, and the volume is written on the
calibrated output grid keyed by the name, as in the winning submission. Finished volumes are backed
up to /checkpoint, so a run stopped by SIGTERM (timeout or preemption) resumes where it left off.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

MODALITIES = {"t1w": "T1w", "t2w": "T2w", "flair": "FLAIR", "swi": "SWI"}
PLANES = {"axi": "AXIAL", "sag": "SAGITTAL", "cor": "CORONAL", "obl": "OBLIQUE"}
NAME_PATTERN = re.compile(r"[-_](?P<modality>[a-z0-9]+)-raw-(?P<plane>[a-z]+)(?:-\d+)?$", re.IGNORECASE)


def series_of(name: str) -> tuple[str, str]:
    """`(modality, plane)` encoded in an `input_image_name`."""
    match = NAME_PATTERN.search(name)
    modality = MODALITIES.get(match.group("modality").lower()) if match else None
    plane = PLANES.get(match.group("plane").lower()) if match else None
    if modality is None or plane is None:
        raise ValueError(f"{name!r} does not end in _<t1w|t2w|flair|swi>-raw-<axi|sag|cor|obl>[-N]")
    return modality, plane


def read_prompts(path: Path) -> list[tuple[str, str]]:
    prompts = json.loads(path.read_text())
    if not isinstance(prompts, list):
        raise ValueError(f"{path} must contain a JSON list")
    cases = []
    for index, item in enumerate(prompts):
        name = str(item.get("input_image_name") or "").strip() if isinstance(item, dict) else ""
        if not name:
            raise ValueError(f"{path} entry {index} has no input_image_name")
        name = name.removesuffix(".nii.gz")
        series_of(name)
        cases.append((name, str(item.get("report") or "")))
    names = [name for name, _ in cases]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"duplicate input_image_name in {path}: {duplicates[:5]}")
    return cases


class Progress:
    """Finished outputs, backed up to the checkpoint directory (the only one that persists)."""

    def __init__(self, output_dir: Path, checkpoint_dir: Path) -> None:
        self.output_dir, self.backup_dir = output_dir, checkpoint_dir / "outputs"
        self.index = checkpoint_dir / "done.json"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.done = set()
        for filename in json.loads(self.index.read_text()) if self.index.exists() else []:
            if (self.backup_dir / filename).exists():
                shutil.copy2(self.backup_dir / filename, output_dir / filename)
                self.done.add(filename)

    def add(self, filename: str) -> None:
        shutil.copy2(self.output_dir / filename, self.backup_dir / filename)
        self.done.add(filename)
        self.save()

    def save(self) -> None:
        tmp = self.index.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(self.done)))
        os.replace(tmp, self.index)


def run(args, pipeline=None) -> int:
    from report2mri.sampling import Report2MRIPipeline, case_seed, save_nifti

    started = time.time()
    cases = read_prompts(args.input / "prompts.json")
    args.output.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.output, args.checkpoint)
    print(f"{len(cases)} prompts, {len(progress.done)} already done", flush=True)

    def stop(signum, _frame):
        progress.save()
        print(f"signal {signum}: saved progress ({len(progress.done)} done)", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    if pipeline is None:
        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("no GPU is visible to the container; pass --device cpu to run on CPU")
        pipeline = Report2MRIPipeline.from_pretrained(args.weights, device=args.device)

    for index, (name, report) in enumerate(cases, start=1):
        filename = f"{name}.nii.gz"
        if filename in progress.done:
            continue
        case_started = time.time()
        modality, plane = series_of(name)
        result = pipeline.generate(report, modality=modality, plane=plane, seed=case_seed(args.seed, name),
                                   case_key=name)
        save_nifti(result.volume.astype(np.float32), result.spacing_mm, args.output / filename)
        progress.add(filename)
        print(f"[{index}/{len(cases)}] {name} {modality} {plane} {result.volume.shape} "
              f"{time.time() - case_started:.1f}s", flush=True)

    missing = [name for name, _ in cases if not (args.output / f"{name}.nii.gz").exists()]
    print(f"wrote {len(cases) - len(missing)}/{len(cases)} volumes in {(time.time() - started) / 60:.1f} min",
          flush=True)
    if missing:
        print(f"ERROR: no volume for {missing[:5]}", flush=True)
        return 1
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, default=Path("/input"))
    p.add_argument("--output", type=Path, default=Path("/output"))
    p.add_argument("--checkpoint", type=Path, default=Path("/checkpoint"))
    p.add_argument("--weights", type=Path,
                   default=Path(os.environ.get("FORITHMUS_WEIGHTS_DIR") or "/weights"))
    p.add_argument("--seed", type=int, default=1234, help="each case uses a seed derived from this and its name")
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args()))
