# Running stewbench on the NVIDIA workstation (Docker + NGC)

Your lab asked you to run models only inside Docker, using an image from
NVIDIA GPU Cloud (NGC). This guide follows those steps.

## 0. One-time checks

```bash
nvidia-smi                 # note "Driver Version" and the GPU name
docker --version
docker run --rm --gpus all nvcr.io/nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi   # GPU visible in Docker?
```

If the last command fails, ask the admins to check the NVIDIA Container Toolkit.

## 1. NGC account and login

1. Create an account at https://ngc.nvidia.com.
2. Profile menu -> **Setup** -> **Generate API Key**.
3. Log in to the registry (the username is literally `$oauthtoken`):

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <your API key>
```

## 2. Pick and pull the PyTorch image

The image's CUDA version must be supported by the workstation driver:

| `nvidia-smi` driver | NGC tag to use |
|---|---|
| >= 570 | `25.01-py3` (default in this repo, CUDA 12.8) |
| >= 580 | `25.09-py3` or newer (CUDA 13.x) |
| < 570 | `24.12-py3` or older |

Check the exact requirement in the release notes:
https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/

```bash
docker pull nvcr.io/nvidia/pytorch:25.01-py3     # ~10 GB, pull once
```

## 3. Get the code and data

```bash
git clone https://github.com/aryankalra404/research-eeg.git
cd research-eeg
git checkout claude/wizardly-einstein-11gv8x     # until merged
# Copy the STEW files so that data/raw/stew/sub01_lo.txt ... sub48_hi.txt and ratings.txt exist
```

## 4. Build the project image (recommended)

This is steps 4-5 of the email ("install software, commit the container") done
reproducibly: the Dockerfile installs our packages on top of the NGC image.

```bash
NGC_TAG=25.01-py3 docker/run.sh build
docker/run.sh check      # prints torch/CUDA/GPU and verifies all 48 subjects
docker/run.sh smoke      # runs the whole pipeline on fake data (~minutes on GPU)
docker/run.sh test       # unit tests
```

`docker/run.sh` mounts `data/`, `outputs/`, `src/` and `configs/` from the host,
so results are written to your normal folders and code edits need no rebuild.

### Alternative: the "docker commit" workflow from the email

```bash
docker run -it --gpus all --ipc=host --name stew-dev \
  -v "$PWD:/workspace/research-eeg" nvcr.io/nvidia/pytorch:25.01-py3 bash
# inside the container:
cd /workspace/research-eeg && pip install -r requirements.txt && pip install --no-deps -e .
exit
docker commit stew-dev stewbench:latest      # reuse later: IMAGE=stewbench:latest docker/run.sh ...
```

## 5. Run the experiments

```bash
docker/run.sh benchmark --config configs/benchmark_quick.yaml      # 1 seed sanity run on real data

# Long runs: detach so they survive closing SSH
DETACH=1 docker/run.sh benchmark --config configs/benchmark.yaml
docker logs -f stewbench          # Ctrl+C stops following, not the run

DETACH=1 docker/run.sh augment --config configs/augmentation.yaml
docker/run.sh neuro --config configs/benchmark.yaml
docker/run.sh benchmark --config configs/benchmark_loso.yaml       # optional, literature comparison
```

Runs are resumable: if the machine reboots, run the same command again.
To use one specific GPU: `GPUS='"device=1"' docker/run.sh ...`.

## 6. Where results go

```
outputs/stew/benchmark_main/report/REPORT.md          summary
outputs/stew/benchmark_main/report/tables/*.tex       paste into the paper
outputs/stew/benchmark_main/report/figures/*.pdf      paper figures
outputs/stew/augmentation_study/report/...
outputs/stew/neurophysiology/figures/...
```

Before a final run, commit your code so each run's `manifest.json` records a
clean git revision (`git_dirty: false`).
