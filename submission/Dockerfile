# Build from the repository root:  docker build -f submission/Dockerfile -t report2mri-submission .
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# The versions the released model was verified with.
RUN pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126 \
    && pip install monai==1.6.0 transformers==5.14.1 tokenizers==0.22.2 einops==0.8.2 safetensors==0.8.0 \
       huggingface_hub==1.26.0 nibabel==5.4.2 numpy==2.5.2 scipy==1.18.0

COPY pyproject.toml README.md LICENSE /opt/report2mri/
COPY report2mri /opt/report2mri/report2mri
RUN pip install --no-deps /opt/report2mri
COPY submission/predict.py /opt/report2mri/predict.py

# The platform runs offline and requires a non-root user.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
RUN useradd --create-home --uid 1000 runner
USER runner
WORKDIR /home/runner
ENTRYPOINT ["python", "/opt/report2mri/predict.py"]
