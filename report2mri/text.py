"""Report text -> conditioning tokens.

Each of three report sections is encoded independently by three frozen encoders. Each encoder's
last hidden state is mean-pooled over real tokens, and the three pooled vectors are concatenated:

    sections  (sequence axis)  findings | impression | acquisition
    features  (feature axis)   MedEmbed-large 1024 | Bio_ClinicalBERT 768 | CXR-BERT 768 = 2560

The result is `(B, 3, 2560)` plus a `(B, 3)` mask that is False for an empty section. The
`acquisition` section is not part of the report: it is `[MODALITY] m [PLANE] p [SPACING] x y z`,
built from the requested series.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

SECTIONS = ("findings", "impression", "acquisition")
MAX_TOKENS = 512


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    hf_repo: str
    revision: str
    directory: str
    weights_file: str
    # Load as a stock BERT instead of through AutoModel. CXR-BERT declares a custom model type whose
    # repository code would otherwise have to be executed; its weights are plain BERT weights.
    stock_bert: bool = False


# Order fixes the feature axis of the conditioning and is part of the checkpoint identity.
ENCODERS = (
    EncoderSpec("medembed_large", "abhinand/MedEmbed-large-v0.1",
                "963121bfb9c625475f65b08fb54990ce9c4e7a1a", "MedEmbed-large-v0.1", "model.safetensors"),
    EncoderSpec("bio_clinicalbert", "emilyalsentzer/Bio_ClinicalBERT",
                "d5892b39a4adaed74b92212a44081509db72f87b", "Bio_ClinicalBERT", "pytorch_model.bin"),
    EncoderSpec("cxr_bert", "microsoft/BiomedVLP-CXR-BERT-specialized",
                "5157bdba1437a3aed316dacb1a5b68edf96b9902", "BiomedVLP-CXR-BERT-specialized",
                "model.safetensors", stock_bert=True),
)

# Checkpoint heads that the encoder-only model does not use: MLM/NSP heads, BERT poolers and
# CXR-BERT's contrastive projection. Any other unexpected or missing weight is an error.
_UNUSED_WEIGHTS = re.compile(r"^(cls\.|cls_projection_head\.|(bert\.)?pooler\.)")


def acquisition_text(modality: str | None, plane: str | None, spacing_mm_xyz: Sequence[float] | None) -> str:
    parts = []
    if modality:
        parts.append(f"[MODALITY] {modality}")
    if plane:
        parts.append(f"[PLANE] {plane}")
    if spacing_mm_xyz is not None:
        if len(tuple(spacing_mm_xyz)) != 3:
            raise ValueError(f"spacing must be 3 values (X, Y, Z), got {tuple(spacing_mm_xyz)!r}")
        parts.append("[SPACING] " + " ".join(f"{float(v):.2f}" for v in spacing_mm_xyz))
    return " ".join(parts)


_SECTION_HEADINGS = {
    "clinical_information": r"clinical(?:\s+information|\s+history)?|history|indication",
    "technique": r"technique|protocol",
    "findings": r"findings?|description",
    "impression": r"impression|conclusion|summary|assessment",
}


def split_report(report: str) -> dict[str, str]:
    """Split a free-text report into sections by `Heading:` markers. Text with no recognised
    heading is treated as findings."""
    text = str(report or "").strip()
    if not text:
        return {}
    pattern = "|".join(f"(?P<{name}>{alts})" for name, alts in _SECTION_HEADINGS.items())
    matches = list(re.finditer(rf"(?<![A-Za-z])(?:{pattern})\s*:", text, re.IGNORECASE))
    if not matches:
        return {"findings": text}
    sections: dict[str, str] = {}
    for position, match in enumerate(matches):
        name = match.lastgroup
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        if body:
            sections[name] = f"{sections[name]} {body}".strip() if name in sections else body
    preamble = text[: matches[0].start()].strip().rstrip(":").strip()
    if preamble:
        sections["clinical_information"] = f"{preamble} {sections.get('clinical_information', '')}".strip()
    return sections or {"findings": text}


def conditioning_sections(findings: str | None, impression: str | None, modality: str, plane: str,
                          spacing_mm_xyz: Sequence[float]) -> dict[str, str]:
    return {
        "findings": (findings or "").strip(),
        "impression": (impression or "").strip(),
        "acquisition": acquisition_text(modality, plane, spacing_mm_xyz),
    }


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(tokens.dtype)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


class FrozenEncoder(nn.Module):
    """One frozen HuggingFace encoder; `encode` returns masked-mean pooled `(B, hidden)`."""

    def __init__(self, spec: EncoderSpec, directory: str | Path, max_tokens: int = MAX_TOKENS) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel, AutoTokenizer, BertConfig, BertModel, BertTokenizerFast
        from transformers.utils import logging as hf_logging

        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(
                f"text encoder '{spec.name}' not found at {directory}. Download it with "
                f"`hf download {spec.hf_repo} --local-dir {directory}`.")
        self.spec = spec
        # transformers' own load report lists the unused heads; they are checked explicitly below.
        verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            if spec.stock_bert:
                self.tokenizer = BertTokenizerFast.from_pretrained(directory, local_files_only=True)
                config = BertConfig.from_pretrained(directory, local_files_only=True)
                self.model, info = BertModel.from_pretrained(directory, config=config, local_files_only=True,
                                                             add_pooling_layer=False, output_loading_info=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True,
                                                               trust_remote_code=False)
                config = AutoConfig.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
                self.model, info = AutoModel.from_pretrained(directory, config=config, local_files_only=True,
                                                             trust_remote_code=False, add_pooling_layer=False,
                                                             output_loading_info=True)
        finally:
            hf_logging.set_verbosity(verbosity)
        # Bio_ClinicalBERT's pytorch_model.bin stores its linear weights transposed. Contiguous
        # weights keep the numerics identical to the safetensors copy the model was trained with.
        for parameter in self.model.parameters():
            parameter.data = parameter.data.contiguous()
        unexpected = [k for k in info["unexpected_keys"] if not _UNUSED_WEIGHTS.match(k)]
        problems = {"missing": list(info["missing_keys"]), "unexpected": unexpected,
                    "mismatched": list(info["mismatched_keys"]), "errors": list(info["error_msgs"])}
        if any(problems.values()):
            raise RuntimeError(f"text encoder '{spec.name}' at {directory} does not match its "
                               f"architecture: { {k: v[:5] for k, v in problems.items() if v} }")
        self.max_tokens = int(max_tokens)
        self.output_dim = int(config.hidden_size)
        self.model.eval().requires_grad_(False)

    def train(self, mode: bool = True):
        return super().train(False)

    @property
    def identity(self) -> dict:
        return {"name": self.spec.name, "hf_repo": self.spec.hf_repo, "output_dim": self.output_dim,
                "max_length": self.max_tokens}

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        batch = self.tokenizer(list(texts), add_special_tokens=True, truncation=True,
                               max_length=self.max_tokens, padding=True, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        tokens = self.model(**batch).last_hidden_state
        return masked_mean(tokens, batch["attention_mask"].to(torch.bool))


class ReportEncoder(nn.Module):
    """The three frozen encoders. `encode(sections)` -> `(tokens (B, 3, D), mask (B, 3))`."""

    def __init__(self, encoders: Sequence[FrozenEncoder]) -> None:
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.output_dim = sum(e.output_dim for e in encoders)

    @classmethod
    def from_directories(cls, directories: Sequence[str | Path], max_tokens: int = MAX_TOKENS) -> "ReportEncoder":
        """Load MedEmbed-large, Bio_ClinicalBERT and CXR-BERT, in that order."""
        if len(directories) != len(ENCODERS):
            raise ValueError(f"expected {len(ENCODERS)} encoder directories, got {len(directories)}")
        return cls([FrozenEncoder(spec, path, max_tokens) for spec, path in zip(ENCODERS, directories)])

    def train(self, mode: bool = True):
        return super().train(False)

    @property
    def identity(self) -> dict:
        """Recorded in adapter checkpoints; loading compares it with the checkpoint's."""
        return {
            "kind": "sectioned_fusion",
            "sections": list(SECTIONS),
            "encoder_order": [e.spec.name for e in self.encoders],
            "encoder_dims": [e.output_dim for e in self.encoders],
            "pooling": "mean",
            "sequence_length": len(SECTIONS),
            "output_dim": self.output_dim,
            "members": [e.identity for e in self.encoders],
        }

    @torch.no_grad()
    def encode(self, sections: Sequence[Mapping[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
        if not sections:
            raise ValueError("encode() received an empty batch")
        tokens, present = [], []
        for name in SECTIONS:
            texts = [str(item.get(name) or "").strip() for item in sections]
            tokens.append(torch.cat([e.encode(texts) for e in self.encoders], dim=-1).unsqueeze(1))
            present.append([bool(t) for t in texts])
        tokens = torch.cat(tokens, dim=1)
        mask = torch.tensor(present, dtype=torch.bool, device=tokens.device).t().contiguous()
        return tokens, mask
