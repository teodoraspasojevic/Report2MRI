"""Generate brain MRI volumes from radiology reports."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from ..geometry import MODALITIES, PLANES
from ..sampling import DEFAULT_MODALITY_GUIDANCE, DEFAULT_REPORT_GUIDANCE, DEFAULT_SEED, DEFAULT_STEPS

log = logging.getLogger("report2mri")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="report2mri-generate",
        description="Generate a 3D brain MRI volume (NIfTI) from a radiology report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Single case: --report/--report-file or --findings/--impression, with --modality, --plane "
               "and --out. Batch: --cases FILE.jsonl with --out-dir; each line is "
               '{"id": ..., "report": ... | "findings": ..., "impression": ..., "modality": ..., "plane": ...}.',
    )
    p.add_argument("--weights", default="weights",
                   help="weights directory (see README), or a Hugging Face model id")
    text = p.add_argument_group("report (single case)")
    text.add_argument("--report", help="free-text report; split on 'Findings:' / 'Impression:' headings")
    text.add_argument("--report-file", type=Path, help="file containing the free-text report")
    text.add_argument("--findings", help="findings section (instead of --report)")
    text.add_argument("--impression", help="impression section (instead of --report)")
    text.add_argument("--modality", choices=MODALITIES)
    text.add_argument("--plane", choices=PLANES)
    text.add_argument("--out", type=Path, help="output .nii.gz")
    batch = p.add_argument_group("batch")
    batch.add_argument("--cases", type=Path, help="JSONL file of cases")
    batch.add_argument("--out-dir", type=Path, help="output directory for --cases")
    sampler = p.add_argument_group("sampling")
    sampler.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="rectified-flow steps")
    sampler.add_argument("--report-guidance-scale", type=float, default=DEFAULT_REPORT_GUIDANCE)
    sampler.add_argument("--modality-guidance-scale", type=float, default=DEFAULT_MODALITY_GUIDANCE)
    sampler.add_argument("--seed", type=int, default=DEFAULT_SEED,
                         help="with --cases, each case uses a seed derived from this and its id")
    sampler.add_argument("--no-grid-calibration", dest="calibrate", action="store_false",
                         help="write the volume on the model's own grid instead of a calibrated acquisition grid")
    sampler.add_argument("--device", default=None, help="e.g. cuda, cuda:1 or cpu (default: cuda if available)")
    args = p.parse_args(argv)

    single = any(v is not None for v in (args.report, args.report_file, args.findings, args.impression))
    if args.cases is not None:
        if single or args.out is not None or args.out_dir is None:
            p.error("--cases needs --out-dir and no single-case report options")
    else:
        if sum(v is not None for v in (args.report, args.report_file)) > 1:
            p.error("give only one of --report and --report-file")
        if (args.report is not None or args.report_file is not None) and (args.findings or args.impression):
            p.error("give either --report/--report-file or --findings/--impression")
        missing = [f"--{n}" for n in ("modality", "plane", "out") if getattr(args, n) is None]
        if not single:
            missing.insert(0, "--report/--findings/--impression or --cases")
        if missing:
            p.error("missing " + ", ".join(missing))
    return args


def read_cases(path: Path) -> list[dict]:
    cases = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        case = json.loads(line)
        missing = [k for k in ("id", "modality", "plane") if not case.get(k)]
        if missing:
            raise SystemExit(f"{path}:{number} is missing {missing}")
        if case["modality"] not in MODALITIES or case["plane"] not in PLANES:
            raise SystemExit(f"{path}:{number}: modality must be one of {MODALITIES}, plane one of {PLANES}")
        if "report" in case and ("findings" in case or "impression" in case):
            raise SystemExit(f"{path}:{number}: give either 'report' or 'findings'/'impression'")
        cases.append(case)
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path} contains duplicate ids")
    return cases


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if args.cases is not None:
        cases = read_cases(args.cases)
        outputs = [args.out_dir / f"{case['id']}.nii.gz" for case in cases]
    else:
        report = args.report_file.read_text() if args.report_file is not None else args.report
        case = {"modality": args.modality, "plane": args.plane}
        case.update({"report": report} if report is not None else {"findings": args.findings, "impression": args.impression})
        cases, outputs = [case], [args.out]

    from ..sampling import Report2MRIPipeline, case_seed, save_nifti

    pipeline = Report2MRIPipeline.from_pretrained(args.weights, device=args.device)
    log.info("loaded weights from %s on %s", args.weights, pipeline.device)
    for case, out in zip(cases, outputs):
        started = time.time()
        seed = case_seed(args.seed, case["id"]) if args.cases is not None else args.seed
        result = pipeline.generate(
            case.get("report"), findings=case.get("findings"), impression=case.get("impression"),
            modality=case["modality"], plane=case["plane"], seed=seed, steps=args.steps,
            report_guidance=args.report_guidance_scale, modality_guidance=args.modality_guidance_scale,
            calibrate=args.calibrate, case_key=out.name.removesuffix(".nii.gz"),
        )
        save_nifti(result.volume.astype(np.float32), result.spacing_mm, out)
        log.info("wrote %s shape=%s spacing=%s mm (%.1fs)", out, result.volume.shape,
                 tuple(round(s, 3) for s in result.spacing_mm), time.time() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
