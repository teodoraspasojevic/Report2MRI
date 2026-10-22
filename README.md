# Report2MRI

**Report2MRI: 3D rectified-flow brain MRI generation conditioned on fused radiology-report embeddings from multiple frozen medical text encoders.**
> 🏆 Winner of the [VLM3D @ MICCAI 2026](https://research.forithmus.com/challenges/mr-volume-generation) MRI Report-to-Volume challenge

[Paper (coming soon)] | [Checkpoint (Hugging Face)](https://huggingface.co/teodora-spasojevic-fau/Report2MRI) | [Challenge](https://research.forithmus.com/challenges/mr-volume-generation)

Report2MRI generates a 3D brain MRI volume (T1w, T2w, FLAIR or SWI; axial, sagittal or coronal) from a free-text radiology report. It leverages NVIDIA's [NV-Generate-MR-Brain](https://huggingface.co/nvidia/NV-Generate-MR-Brain) latent diffusion model frozen and adds five zero-initialised cross-attention adapters.

## Overview

**Report conditioning**: Following [Report2CT](https://arxiv.org/abs/2509.14780), the findings and impression sections are encoded separately by three frozen medical text encoders (MedEmbed-large, Bio_ClinicalBERT, CXR-BERT), mean-pooled and concatenated into one 2560-d token per section. We add a third token describing the requested series as text (`[MODALITY] m [PLANE] p [SPACING] x y z`).

**Training**: The backbone's own objective, an L1 loss on the rectified-flow velocity, trains only the adapters and a context projection. The UNet, VAE and text encoders stay frozen. Report and modality labels are each dropped with p = 0.1 so that classifier-free guidance can be used at inference.

**Inference**: 30 rectified-flow steps with hierarchical classifier-free guidance, in which the report is applied on top of the base model's modality guidance (modality scale 7, report scale 3). The volume is then resampled to a slice thickness and in-plane spacing drawn from the training-split acquisition statistics for its modality and plane, keeping the physical field of view. This output-grid calibration matters because the challenge's 2.5D FID slices each volume on the grid it is written on.

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

How the submitted configuration was built up (official test-set scores):

| Configuration | SSIM ↑ | PSNR ↑ | MSE ↓ | FID-2.5D ↓ |
|---|---|---|---|---|
| Report adapters, no grid calibration | 0.4420 | 13.50 | 0.0461 | 344.29 |
| + slice-axis resampling | 0.4420 | 13.57 | 0.0453 | 215.96 |
| + modality guidance 10 → 7 | 0.4473 | 13.64 | 0.0446 | 206.63 |
| + in-plane resampling (**submitted**) | 0.4437 | 13.64 | 0.0447 | **204.05** |

Run-to-run spread of the same container is up to 0.002 SSIM and 4.9 FID, so the last row is within noise of the one above.

## Pretrained Checkpoint

The trained adapter is on Hugging Face:

👉 [teodora-spasojevic-fau/Report2MRI](https://huggingface.co/teodora-spasojevic-fau/Report2MRI)

The base model and text encoders are not re-hosted; download them from their original sources. Collect everything into one `weights/` directory:

```bash
mkdir -p weights && cd weights

# Report2MRI adapter (adapter.pt, arm.txt, ...)
hf download teodora-spasojevic-fau/Report2MRI --local-dir .

# Frozen NVIDIA base UNet (use v0, not v1) and MAISI VAE
hf download nvidia/NV-Generate-MR-Brain models/diff_unet_3d_rflow-mr-brain_v0.pt --local-dir .
hf download nvidia/NV-Generate-CT models/autoencoder_v1.pt --local-dir .
mv models/*.pt . && rmdir models

# Frozen text encoders
hf download abhinand/MedEmbed-large-v0.1 --local-dir MedEmbed-large-v0.1
hf download emilyalsentzer/Bio_ClinicalBERT --local-dir Bio_ClinicalBERT
hf download microsoft/BiomedVLP-CXR-BERT-specialized --local-dir BiomedVLP-CXR-BERT-specialized
```

The adapter refuses to load on a base UNet whose sha256 differs from the one recorded in the checkpoint.

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
  author       = {Spasojevic, Teodora and Wang, Jiayi and Durug{\"o}l, {\"O}mer Faruk and Hagen, Luca and N{\"u}tzel, Felix},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/teodora-spasojevic-fau/Report2MRI}},
  note         = {1st place, VLM3D @ MICCAI 2026, MRI Report-to-Volume}
}
```
