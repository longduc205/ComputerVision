# Robust Object Detection in Satellite Images under Distribution Shift

Research framework for training and evaluating object detectors across climate zones and disaster scenarios using RWDS benchmark datasets.

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM for training)
- **RAM**: 16GB+ recommended
- **Disk**: 50GB+ for datasets and checkpoints
- **OS**: Linux (Ubuntu 20.04+), Windows (WSL2), or macOS

Check your GPU and CUDA version:

```bash
nvidia-smi          # Check GPU model and CUDA driver version
nvcc --version      # Check CUDA toolkit version
```

## Software Requirements

- **Python**: 3.10 or 3.11 (recommended), 3.12 supported
- **CUDA**: 11.8 or 12.x (if using GPU)
- **Git**: Required for installing detectron2 and remote-clip from GitHub

## Environment Setup

### Step 1: Clone the repository

```bash
git clone <repository-url>
cd ComputerVision
```

### Step 2: Create a virtual environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it (Linux/macOS/WSL2)
source venv/bin/activate

# On Windows (Command Prompt)
# venv\Scripts\activate.bat

# On Windows (PowerShell)
# venv\Scripts\Activate.ps1
```

### Step 3: Install PyTorch (choose one based on your CUDA version)

First, activate your virtual environment, then run one of:

```bash
# Option A: CUDA 12.1 (recommended for RTX 30/40, A100, H100, or unknown)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Option B: CUDA 11.8 (for older GPUs like RTX 20 series, V100)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Option C: CPU only (no GPU, training will be very slow)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Install the project and remaining dependencies

```bash
pip install -e ".[dev]"
```

### Step 5: Install detectron2 and remote-clip (optional)

These packages require torch to be available during build, so they must be installed **after** torch is in the environment and **without** pip's build isolation:

```bash
# detectron2 - requires torch already installed
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation

# remote-clip - requires torch already installed
pip install 'git+https://github.com/ContextNet/remoteclip.git' --no-build-isolation
```

If you skip this step, the core training/evaluation pipeline will still work.

### Step 6: Verify installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import mmdet; print(f'mmdet {mmdet.__version__}')"
python -c "import detectron2; print('detectron2 OK')"
```

Expected output (with GPU):
```
PyTorch 2.x.x, CUDA: True
mmdet 3.x.x
detectron2 OK
```

## Quick Start

### Download datasets

```bash
python scripts/download_data.py --dataset xview --output data/raw/
```

### Preprocess

```bash
python scripts/preprocess_data.py --input data/raw/xview --output data/processed/
```

### Train

```bash
python scripts/train.py --config configs/rwds_cz/single_source/tropical.yaml --gpu 0
```

### Evaluate

```bash
python scripts/evaluate.py --checkpoint results/experiments/exp_id/checkpoint_best.pth --domains tropical arid temperate
```

## Project Structure

See `docs/superpowers/specs/` for architecture design.

## Citation

Coming soon.
