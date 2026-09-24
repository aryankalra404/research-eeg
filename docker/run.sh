#!/usr/bin/env bash
# Helper for running the STEW benchmark inside the NGC-based container.
#
#   docker/run.sh build            build the image (NGC_TAG=25.01-py3 by default)
#   docker/run.sh check            GPU + dataset check
#   docker/run.sh smoke            full pipeline on a tiny synthetic fixture
#   docker/run.sh test             unit tests
#   docker/run.sh <any stewbench command...>
#                                  e.g. docker/run.sh benchmark --config configs/benchmark.yaml
#   docker/run.sh shell            interactive bash inside the container
#
# Long runs: DETACH=1 docker/run.sh benchmark --config configs/benchmark.yaml
#            docker logs -f stewbench      (safe to close the SSH session)
set -euo pipefail

IMAGE="${IMAGE:-stewbench:latest}"
NGC_TAG="${NGC_TAG:-25.01-py3}"
GPUS="${GPUS:-all}"              # e.g. GPUS='"device=1"' to pin one GPU
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mounts=(
  -v "$ROOT/data:/workspace/research-eeg/data"
  -v "$ROOT/outputs:/workspace/research-eeg/outputs"
  -v "$ROOT/src:/workspace/research-eeg/src"          # live code: no rebuild after edits
  -v "$ROOT/configs:/workspace/research-eeg/configs"
  -v "$ROOT/tests:/workspace/research-eeg/tests"
)
common=(--gpus "$GPUS" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
        --user "$(id -u):$(id -g)" -e HOME=/tmp -w /workspace/research-eeg "${mounts[@]}")

run() {
  if [[ "${DETACH:-0}" == "1" ]]; then
    docker rm -f stewbench >/dev/null 2>&1 || true
    docker run -d --name stewbench "${common[@]}" "$IMAGE" "$@"
    echo "Started. Follow with: docker logs -f stewbench"
  else
    docker run --rm -it "${common[@]}" "$IMAGE" "$@"
  fi
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  build) docker build -f "$ROOT/docker/Dockerfile" --build-arg "NGC_TAG=$NGC_TAG" -t "$IMAGE" "$ROOT" ;;
  shell) run bash ;;
  test)  run python -m pytest -q tests ;;
  help)  sed -n '2,15p' "$0" ;;
  *)     run python -m stewbench "$cmd" "$@" ;;
esac
