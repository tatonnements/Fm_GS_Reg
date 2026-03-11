# Fm_GS_Reg

Functional Map-based registration of two 3D Gaussian Splatting submaps.

## Pipeline

1. Load two 3D Gaussian submaps (agent0, agent1)
2. Compute LBO eigenbasis for each using the Gaussian Laplacian
3. Compute spectral descriptors (HKS/WKS) and build functional map
4. Convert functional map to point-to-point correspondences
5. Estimate rigid transformation (R, t) from correspondences with RANSAC
6. Compare with ground truth

## Prerequisites

- Linux (tested on Ubuntu)
- NVIDIA GPU with CUDA 12.1+ support
- [Conda](https://docs.conda.io/en/latest/miniconda.html) package manager
- C++ compiler and CMake ≥ 3.22

## Setup

### 1. Clone the repository

```bash
git clone --recursive <repo-url>
cd Fm_GS_Reg
```

### 2. Create the conda environment

```bash
conda create -n fm_gs python=3.8 cmake>=3.22 gmp cgal -c conda-forge -y
conda activate fm_gs
```

### 3. Install PyTorch with CUDA support

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Install third-party CUDA submodules for 3DGaussianLaplacian

These packages contain custom CUDA extensions and must be built from source:

```bash
pip install third_party/3DGaussianLaplacian/submodules/diff-gaussian-rasterization
pip install third_party/3DGaussianLaplacian/submodules/simple-knn
pip install third_party/3DGaussianLaplacian/submodules/robust-laplacian-gs
pip install third_party/3DGaussianLaplacian/submodules/tetra-triangulation
```

### 6. Install pyFM (Functional Maps)

```bash
pip install pyfmaps
```

## Usage

```bash
python run_fmaps_registration.py
```

## Project Structure

```
Fm_GS_Reg/
├── run_fmaps_registration.py   # Main registration pipeline
├── transform_agent0.py         # Transform utility for agent0
├── requirements.txt            # Python dependencies
├── src/                        # Source modules
│   ├── fmaps_model.py
│   ├── gaussian_model.py
│   ├── gaussian_model_utils.py
│   └── transform_gaussian.py
├── data/                       # Data directory
│   └── toycase/                # Example data
└── third_party/
    ├── 3DGaussianLaplacian/    # Gaussian Laplacian (with CUDA submodules)
    └── pyFM/                   # Functional Maps library
```
