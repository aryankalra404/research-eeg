# STEW benchmark image on top of NVIDIA's NGC PyTorch container.
#
# Pick the NGC tag that matches the workstation's NVIDIA driver (run `nvidia-smi`):
#   driver >= 570  ->  25.01-py3  (CUDA 12.8)
#   Verified on the VIT RTX A4000 workstation (driver 580): 26.07-py3 (default, CUDA forward-compat)
#   driver >= 580  ->  25.09-py3 or newer (CUDA 13.x)
#   driver  < 570  ->  24.12-py3 or older (check the NGC release notes)
# Build:  docker build -f docker/Dockerfile -t stewbench:latest --build-arg NGC_TAG=25.01-py3 .
ARG NGC_TAG=26.07-py3
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

WORKDIR /workspace/research-eeg

# Dependencies first (cached layer). torch comes from the NGC base image.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY tests ./tests
RUN pip install --no-cache-dir --no-deps -e . \
 && python -c "import torch, stewbench; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# Data, outputs and (optionally) code are bind-mounted at run time; see docker/run.sh.
CMD ["python", "-m", "stewbench", "--help"]
