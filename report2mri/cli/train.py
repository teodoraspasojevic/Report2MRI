"""Train the report adapters on a frozen NV-Generate-MR-Brain UNet.

Single GPU:  report2mri-train --manifest train.jsonl --weights weights --out runs/my_run
Multi-GPU:   torchrun --nproc_per_node=4 -m report2mri.cli.train --manifest ... (same flags)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

log = logging.getLogger("report2mri")


def parse_args(argv=None):
    from ..training import TrainingConfig

    d = TrainingConfig()
    p = argparse.ArgumentParser(
        prog="report2mri-train",
        description="Train the Report2MRI adapters. The base UNet, VAE and text encoders stay frozen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='--manifest is a JSONL file, one series per line: {"image": "path.nii.gz", "findings": "...", '
               '"impression": "...", "modality": "T1w|T2w|FLAIR|SWI", "plane": "AXIAL|SAGITTAL|CORONAL|OBLIQUE"}. '
               "The released model used --batch-size 8 --grad-accumulation-steps 4 on 8 GPUs (effective batch 256).",
    )
    p.add_argument("--manifest", type=Path, required=True, help="training manifest (JSONL)")
    p.add_argument("--weights", type=Path, default=Path("weights"),
                   help="directory with the base UNet, VAE and text encoders (see README)")
    p.add_argument("--out", type=Path, required=True, help="output directory for adapter checkpoints")
    p.add_argument("--epochs", type=int, default=d.epochs, help="total number of epochs")
    p.add_argument("--max-steps", type=int, default=None, help="stop after this many micro-steps (smoke tests)")
    p.add_argument("--batch-size", type=int, default=d.batch_size, help="per GPU")
    p.add_argument("--grad-accumulation-steps", type=int, default=d.grad_accumulation_steps)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--grad-clip-norm", type=float, default=d.grad_clip_norm, help="0 disables clipping")
    p.add_argument("--report-dropout", type=float, default=d.report_dropout)
    p.add_argument("--modality-dropout", type=float, default=d.modality_dropout)
    p.add_argument("--no-amp", dest="amp", action="store_false", help="disable bfloat16 autocast")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=d.log_every, help="optimizer steps")
    p.add_argument("--save-every-steps", type=int, default=None, help="optimizer steps between checkpoints")
    p.add_argument("--resume", type=Path, default=None, help="adapter checkpoint written by this command")
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    return p.parse_args(argv)


class ShardedBatchSampler:
    """Every `world_size`-th batch, offset by `rank`; each rank gets the same number of batches."""

    def __init__(self, sampler, rank: int, world_size: int) -> None:
        self.sampler, self.rank, self.world_size = sampler, rank, world_size

    def __len__(self) -> int:
        return len(self.sampler) // self.world_size

    def __iter__(self):
        limit = len(self) * self.world_size
        for i, batch in enumerate(self.sampler):
            if i >= limit:
                break
            if i % self.world_size == self.rank:
                yield batch

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)


def setup_distributed(device_arg):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        device = torch.device(device_arg or ("cuda" if torch.cuda.is_available() else "cpu"))
        return device, 0, 0, 1
    import torch.distributed as dist

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if device_arg == "cpu" or not torch.cuda.is_available():
        backend, device = "gloo", torch.device("cpu")
    else:
        backend, device = "nccl", torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)
    return device, rank, local_rank, world_size


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    from torch.utils.data import DataLoader

    from ..adapter import local_weight_paths, sha256_file
    from ..data import BucketBatchSampler, ReportVolumeDataset, collate
    from ..model import build_unet, freeze_base, load_base_weights, read_base_checkpoint
    from ..nvidia import build_noise_scheduler, load_autoencoder
    from ..text import ReportEncoder
    from ..training import AdapterTrainer, TrainingConfig

    device, rank, local_rank, world_size = setup_distributed(args.device)
    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)
    config = TrainingConfig(
        lr=args.lr, epochs=args.epochs, max_steps=args.max_steps, batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accumulation_steps, grad_clip_norm=args.grad_clip_norm or None,
        amp=args.amp, report_dropout=args.report_dropout, modality_dropout=args.modality_dropout,
        seed=args.seed, log_every=args.log_every, save_every_steps=args.save_every_steps,
    )

    dataset = ReportVolumeDataset(args.manifest)
    sampler = BucketBatchSampler(dataset, batch_size=args.batch_size, seed=args.seed)
    if world_size > 1:
        sampler = ShardedBatchSampler(sampler, rank, world_size)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    log.info("%d series in %d (modality, plane) buckets, %d batches per epoch per rank",
             len(dataset), len(getattr(sampler, "sampler", sampler).buckets), len(sampler))

    paths = local_weight_paths(args.weights, require_adapter=False)
    text_encoder = ReportEncoder.from_directories(paths.text_encoders).to(device)
    base_state, scale_factor = read_base_checkpoint(paths.base_unet)
    unet = build_unet(use_flash_attention=device.type == "cuda")
    load_base_weights(unet, base_state)
    del base_state
    freeze_base(unet)
    unet = unet.to(device)
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel

        unet = DistributedDataParallel(unet, device_ids=[local_rank] if device.type == "cuda" else None,
                                       find_unused_parameters=True)

    trainer = AdapterTrainer(unet, text_encoder, load_autoencoder(paths.vae, device=device), build_noise_scheduler(),
                             scale_factor, sha256_file(paths.base_unet), config, device, args.out, rank=rank)
    if args.resume:
        trainer.resume(args.resume)
    summary = trainer.fit(loader)
    if rank == 0:
        (args.out / "train_summary.json").write_text(json.dumps(summary, indent=2))
        log.info("done: %s", summary)
    if world_size > 1:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
