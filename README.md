# Report2MRI

**Report2MRI: 3D rectified-flow brain MRI generation conditioned on fused radiology-report embeddings from multiple frozen medical text encoders.**
> 🏆 Winner of the [VLM3D @ MICCAI 2026](https://research.forithmus.com/challenges/mr-volume-generation) MRI Report-to-Volume challenge

[Paper (coming soon)] | [Checkpoint (Hugging Face)](https://huggingface.co/teodora-spasojevic-fau/Report2MRI) | [Challenge](https://research.forithmus.com/challenges/mr-volume-generation)

Report2MRI generates a 3D brain MRI volume (T1w, T2w, FLAIR or SWI; axial, sagittal or coronal) from a free-text radiology report. It leverages NVIDIA's [NV-Generate-MR-Brain](https://huggingface.co/nvidia/NV-Generate-MR-Brain) latent diffusion model frozen and adds five zero-initialised cross-attention adapters.

## Overview

**Report conditioning**: Following [Report2CT](https://arxiv.org/abs/2509.14780), the findings and impression sections are encoded separately by three frozen medical text encoders (MedEmbed-large, Bio_ClinicalBERT, CXR-BERT), mean-pooled and concatenated into one 2560-d token per section. We add a third token describing the requested series as text (`[MODALITY] m [PLANE] p [SPACING] x y z`) and embed it in the same manner.

**Training**: The backbone's own objective, an L1 loss on the rectified-flow velocity, trains only the adapters and a context projection. The UNet, VAE and text encoders stay frozen. Report and modality labels are each dropped with p = 0.1 so that classifier-free guidance can be used at inference. The model is trained to generate at [NV-Generate-MR-Brain](https://huggingface.co/nvidia/NV-Generate-MR-Brain)'s recommended FOV with spacing ~1mm.

**Inference**: 30 rectified-flow steps with hierarchical classifier-free guidance, in which the report is applied on top of the base model's modality guidance (modality scale 7, report scale 3). In the final submissions, generated volumes are kept at their training FOV but are resampled to a slice thickness and in-plane spacing drawn from the training-split of MR-RATE.

## Method

```
report ─┬ findings    ┐  each section is encoded separately by three frozen encoders,
        ├ impression  ├─ mean-pooled and concatenated: 1024 (MedEmbed) + 768 (Bio_ClinicalBERT)
        └ acquisition ┘  + 768 (CXR-BERT) = 2560 features per section → context (B, 3, 2560)
                │
          context_proj (trainable): LayerNorm → Linear 2560→2048 → GELU → Linear 2048→512 → LayerNorm
                │
latent ──► NV-Generate-MR-Brain UNet (frozen) + 5 zero-initialised cross-attention adapters (trainable)
                │
          MAISI VAE decoder (frozen) → output-grid calibration → NIfTI volume
```

The adapters sit at the two coarsest encoder levels, the two mirrored decoder levels and the bottleneck. Because they are zero-initialised, the untrained model reproduces NVIDIA's model exactly.

## Results

Official scores on the hidden test set (690 series, all scored). The ranking is the mean rank over the four metrics.

| | SSIM ↑ | PSNR ↑ (dB) | MSE ↓ | FID-2.5D ↓ | Mean rank |
|---|---|---|---|---|---|
| **Report2MRI** | 0.4437 | **13.64** | **0.0447** | **204.05** | **1.25 (1st)** |
| 2nd place | **0.4891** | 13.49 | 0.0466 | 263.51 | 1.75 |
| 3rd place | 0.3716 | 13.00 | 0.0525 | 1190.57 | 3.00 |

How the submitted configuration was built up (official test-set scores):

| Configuration | SSIM ↑ | PSNR ↑ | MSE ↓ | FID-2.5D ↓ |
|---|---|---|---|---|
| Report adapters, no grid calibration | 0.4420 | 13.50 | 0.0461 | 344.29 |
| + slice-axis resampling | 0.4420 | 13.57 | 0.0453 | 215.96 |
| + modality guidance 10 → 7 | 0.4473 | 13.64 | 0.0446 | 206.63 |
| + in-plane resampling (**submitted**) | 0.4437 | 13.64 | 0.0447 | **204.05** |

Run-to-run spread of the same container is up to 0.002 SSIM and 4.9 FID, so the last row is within noise of the one above.

## Installation

Requires Python ≥ 3.10 and PyTorch ≥ 2.6 (tested with Python 3.12, PyTorch 2.13–2.14, MONAI 1.6 and transformers 5.14–5.20). Install a PyTorch build matching your CUDA version first if needed (see [pytorch.org](https://pytorch.org)), then:

```bash
git clone https://github.com/teodoraspasojevic/Report2MRI.git
cd Report2MRI
pip install -e .
```

## Pretrained Checkpoint

The trained adapter is on Hugging Face:

👉 [teodora-spasojevic-fau/Report2MRI](https://huggingface.co/teodora-spasojevic-fau/Report2MRI)

The base model and text encoders are not re-hosted; download them from their original sources. Collect everything into one `weights/` directory:

```bash
mkdir -p weights && cd weights

# Report2MRI adapter and the spacing tables used for output-grid calibration
hf download teodora-spasojevic-fau/Report2MRI adapter.pt native_spacing_table.json inplane_spacing_table.json --local-dir .

# Frozen NVIDIA base UNet (use v0, not v1) and MAISI VAE
hf download nvidia/NV-Generate-MR-Brain models/diff_unet_3d_rflow-mr-brain_v0.pt --local-dir .
hf download nvidia/NV-Generate-CT models/autoencoder_v1.pt --local-dir .
mv models/*.pt . && rmdir models

# Frozen text encoders
hf download abhinand/MedEmbed-large-v0.1 config.json model.safetensors special_tokens_map.json tokenizer.json tokenizer_config.json vocab.txt --local-dir MedEmbed-large-v0.1
hf download emilyalsentzer/Bio_ClinicalBERT config.json pytorch_model.bin vocab.txt --local-dir Bio_ClinicalBERT
hf download microsoft/BiomedVLP-CXR-BERT-specialized config.json model.safetensors special_tokens_map.json tokenizer_config.json vocab.txt --local-dir BiomedVLP-CXR-BERT-specialized
cd ..
```

The adapter refuses to load on a base UNet whose sha256 differs from the one recorded in the checkpoint. Loading is strict: a missing or unexpected adapter or base tensor is an error. The only checkpoint entries the text encoders are allowed to leave unused are their MLM, pooler and (for CXR-BERT) projection heads. No remote code is executed.

## Inference

```bash
report2mri-generate --weights weights \
    --report "Findings: A 12 mm T2-hyperintense lesion in the right frontal white matter without mass effect. Impression: Nonspecific white matter lesion." \
    --modality T2w --plane AXIAL --out outputs/case001.nii.gz
```

The report is split on its `Findings:` and `Impression:` headings; alternatively pass `--findings` and `--impression`. `--modality` is `T1w`, `T2w`, `FLAIR` or `SWI`, and `--plane` is `AXIAL`, `SAGITTAL`, `CORONAL` or `OBLIQUE`. For many reports, pass a JSONL file with one `{"id", "report", "modality", "plane"}` object per line as `--cases`, together with `--out-dir`. From Python:

```python
from report2mri import Report2MRIPipeline, save_nifti

pipeline = Report2MRIPipeline.from_pretrained("weights")   # or a Hugging Face model id
result = pipeline.generate("Findings: ... Impression: ...", modality="T2w", plane="AXIAL", case_key="case001")
save_nifti(result.volume, result.spacing_mm, "case001.nii.gz")
```

The defaults are the submitted configuration: 30 steps, modality guidance 7, report guidance 3 and output-grid calibration. The output is a RAS NIfTI in NVIDIA's MR intensity range. `--no-grid-calibration` keeps the model's ~1 mm grid.

## Training

Training reads a JSONL manifest with one series per line; `image` may be relative to the manifest:

```json
{"image": "path/to/series.nii.gz", "findings": "...", "impression": "...", "modality": "T2w", "plane": "AXIAL"}
```

```bash
report2mri-train --manifest train.jsonl --weights weights --out runs/report2mri
# multi-GPU
torchrun --nproc_per_node=8 -m report2mri.cli.train --manifest train.jsonl --weights weights --out runs/report2mri
```

Volumes are reoriented to RAS, resampled to their (modality, plane) grid and percentile-normalised. The defaults reproduce the released adapter: Adam at learning rate 1e-3, 2 epochs, and batch size 8 with gradient accumulation 4 per GPU on 8 GPUs. Checkpoints contain only the adapter and can replace `adapter.pt` in a weights directory; `--resume` continues a run. Downloading and preparing MR-RATE is not part of this repository.

## Challenge submission

[`submission/`](submission) holds the challenge container. [`predict.py`](submission/predict.py) turns `/input/prompts.json` into `/output/<input_image_name>.nii.gz` with the submitted settings, reading the modality and plane from each name, and resumes from `/checkpoint` after a SIGTERM. Each case is seeded from its name; the leaderboard run itself was unseeded.

```bash
docker build -f submission/Dockerfile -t report2mri-submission .
submission/make_weights_zip.sh weights weights.zip
submission/package_image.sh report2mri-submission report2mri-submission.tar.gz   # classic docker archive, needs skopeo
forithmus submit report2mri-submission.tar.gz --weights weights.zip --tier gpu-a100-80 --time-budget 200
```

## Submission summary

All official hidden-test results of this model (690 series each), in submission order. Besides the final three-encoder fusion, the text conditioning was also tried with single encoders: CXR-BERT's [CLS] token alone, or all CXR-BERT or RadBERT token states. Grid calibration is the output-grid resampling described in the Overview. The submitted entry is in bold.

| Date | Text conditioning | Report guidance | Modality guidance | Grid calibration | Other change | SSIM ↑ | PSNR ↑ | MSE ↓ | FID-2.5D ↓ |
|---|---|---|---|---|---|---|---|---|---|
| 2026-08-21 | CXR-BERT [CLS] token | 7 | 10 | none | – | 0.4481 | 13.22 | 0.0494 | 309.66 |
| 2026-08-21 | Three-encoder fusion | 3 | 10 | none | – | 0.4420 | 13.50 | 0.0461 | 344.29 |
| 2026-08-21 | CXR-BERT tokens | 4 | 10 | none | – | 0.4443 | 13.48 | 0.0466 | 392.65 |
| 2026-08-21 | RadBERT tokens | 4 | 10 | none | – | 0.4512 | 13.56 | 0.0454 | 406.38 |
| 2026-08-23 | CXR-BERT [CLS] token | 7 | 10 | none | best of 5 seeds per case | 0.4551 | 13.39 | 0.0474 | 308.40 |
| 2026-08-23 | CXR-BERT [CLS] token | 7 | 10 | none | 50 sampling steps | 0.4483 | 13.25 | 0.0491 | 311.28 |
| 2026-08-23 | CXR-BERT [CLS] token | 9 | 10 | none | – | 0.4461 | 13.21 | 0.0498 | 301.19 |
| 2026-08-23 | CXR-BERT [CLS] token | 7 | 10 | slice axis | – | 0.4504 | 13.27 | 0.0492 | 247.72 |
| 2026-08-23 | CXR-BERT [CLS] token | 7 | 10 | none | impression before findings | 0.4511 | 13.26 | 0.0492 | 314.13 |
| 2026-08-24 | CXR-BERT [CLS] token | 8 | 10 | none | – | 0.4465 | 13.23 | 0.0494 | 308.00 |
| 2026-08-24 | CXR-BERT [CLS] token | 7 | 10 | none | posterior-shift correction (first version) | 0.4432 | 13.16 | 0.0503 | 300.90 |
| 2026-08-24 | CXR-BERT [CLS] token | 7 | 10 | none | posterior-shift correction (fixed) | 0.4491 | 13.23 | 0.0494 | 308.81 |
| 2026-08-24 | CXR-BERT [CLS] token | 7 | 7 | none | – | 0.4548 | 13.33 | 0.0482 | 297.63 |
| 2026-08-24 | CXR-BERT [CLS] token | 4 | 10 | none | – | 0.4510 | 13.25 | 0.0492 | 330.84 |
| 2026-08-25 | CXR-BERT [CLS] token | 9 | 10 | slice axis | – | 0.4454 | 13.21 | 0.0497 | 262.92 |
| 2026-08-25 | CXR-BERT [CLS] token | 7 | 10 | slice axis | brain crop | 0.4420 | 13.15 | 0.0506 | 253.32 |
| 2026-08-25 | CXR-BERT [CLS] token | 9 | 10 | none | brain crop | 0.4415 | 13.14 | 0.0505 | 298.77 |
| 2026-08-25 | CXR-BERT [CLS] token | 9 | 10 | slice axis | brain crop | 0.4392 | 13.14 | 0.0507 | 255.84 |
| 2026-08-27 | CXR-BERT [CLS] token | 7 | 7 | slice axis | – | 0.4556 | 13.35 | 0.0481 | 240.94 |
| 2026-08-31 | Three-encoder fusion | 3 | 10 | slice axis | – | 0.4428 | 13.56 | 0.0454 | 215.69 |
| 2026-09-01 | CXR-BERT [CLS] token | 7 | 7 | slice axis | Gaussian instead of box slab kernel | 0.4539 | 13.32 | 0.0485 | 252.74 |
| 2026-09-02 | CXR-BERT [CLS] token | 7 | 7 | slice axis + in-plane | – | 0.4522 | 13.32 | 0.0485 | 247.98 |
| 2026-09-02 | Three-encoder fusion | 3 | 10 | slice axis + in-plane | – | 0.4386 | 13.56 | 0.0454 | 212.39 |
| 2026-09-02 | Three-encoder fusion | 3 | 7 | slice axis | – | 0.4483 | 13.64 | 0.0447 | 204.18 |
| **2026-09-05** | **Three-encoder fusion** | **3** | **7** | **slice axis + in-plane** | **–** | **0.4437** | **13.64** | **0.0447** | **204.05** |
| 2026-09-05 | Three-encoder fusion | 3 | 7 | slice axis | background set to 0 | 0.4432 | 13.57 | 0.0453 | 214.81 |

## License

- **Code**: [Apache License 2.0](LICENSE). Builds on [MONAI](https://github.com/Project-MONAI/MONAI) and [NV-Generate-CTMR](https://github.com/nvidia-medtech/NV-Generate-CTMR) (both Apache-2.0).
- **Adapter weights**: CC BY-NC-SA 4.0, as a derivative of MR-RATE.
- **Base model**: NV-Generate-MR-Brain and the MAISI VAE are under the NVIDIA Open Model License and are not redistributed.
- **Data**: [MR-RATE](https://huggingface.co/datasets/Forithmus/MR-RATE), CC BY-NC-SA 4.0.

## Acknowledgements

This work was supported through the Bavarian AI Foundation Model Initiative (AI-Bay.eu), funded by the Bavarian State Ministry of Science and the Arts (StMWK).

The authors gratefully acknowledge the scientific support and HPC resources provided by the Erlangen National High Performance Computing Center (NHR@FAU) of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) under the BayernKI project.

## Citation

If you use this code or model, please cite:

```bibtex
@misc{spasojevic2026report2mri,
  title        = {Report-Conditioned Adapters on a Frozen 3D Brain-MRI Diffusion Prior},
  author       = {Spasojevic, Teodora and Wang, Jiayi and Durugoel, Oemer Faruk and Hagen, Luca and Nuetzel, Felix},
  year         = {2026},
  howpublished = {\url{https://github.com/teodoraspasojevic/Report2MRI}},
  note         = {1st place, VLM3D @ MICCAI 2026, MRI Report-to-Volume}
}
```
