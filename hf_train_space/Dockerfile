# Build from repository root:
#   docker build -f hf_train_space/Dockerfile -t npa-train .
#
# HF Space: set Dockerfile path to hf_train_space/Dockerfile, build context = repo root.

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY hf_train_space/requirements.txt /tmp/space-requirements.txt
RUN pip install --no-cache-dir -r /tmp/space-requirements.txt

# Full project (simulator + agents + train_dqn.py)
COPY . /app

EXPOSE 7860

CMD ["python", "hf_train_space/app.py"]
