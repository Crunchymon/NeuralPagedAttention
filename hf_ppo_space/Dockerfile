# Build from repository root:
#   docker build -f hf_ppo_space/Dockerfile -t npa-ppo-ddp .
#
# Hugging Face Space:
#   - SDK: Docker
#   - Dockerfile path: hf_ppo_space/Dockerfile
#   - Build context: repository root (parent of hf_ppo_space/)
#   - Hardware: 4x Nvidia L4 (or any multi-GPU CUDA box)

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

# Stage Python deps before copying the repo so layer caching is friendly.
COPY hf_ppo_space/requirements.txt /tmp/space-requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/space-requirements.txt

# Whole repo (simulator + agents + new DDP trainer + accelerate config)
COPY . /app

EXPOSE 7860

CMD ["python", "hf_ppo_space/app.py"]
