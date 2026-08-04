# SCG-Mamba Environment

The following environment was collected from the `mamba_clone` Conda environment used for SCG-Mamba. The reported experiments use the VMamba `selective_scan_cuda_core` backend.

## Platform

| Component | Version |
|---|---|
| Operating system | Ubuntu 20.04.1 LTS (Focal Fossa) |
| Linux kernel | 5.15.0-139-generic |
| Architecture | x86_64 |
| glibc | 2.31 |
| Python | 3.10.19 |
| Python build compiler | GCC 11.2.0 |
| System GCC/G++ | 9.4.0 |
| CUDA Toolkit (`nvcc`) | 11.8, V11.8.89 |
| NVIDIA driver | TODO: collect on the GPU training node |
| GPU | TODO: confirm on the GPU training node |

The environment report was collected in a session where the NVIDIA driver was not visible: `torch.cuda.is_available()` returned `False`, and `nvidia-smi` could not communicate with the driver. The GPU model and driver version must therefore be collected again on the node/session actually used for training.

## PyTorch stack

| Package/component | Version |
|---|---|
| PyTorch | 2.3.1+cu118 |
| TorchVision | 0.18.1+cu118 |
| PyTorch CUDA runtime | 11.8 |
| cuDNN | 8.7.0 |
| PyTorch CXX11 ABI | False |
| Triton | 2.3.1 |

## Runtime dependencies

| Package | Version |
|---|---|
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| scikit-learn | 1.7.2 |
| tqdm | 4.67.3 |
| einops | 0.8.2 |
| timm | 1.0.25 |

## Selective-scan build dependencies

| Package | Version |
|---|---|
| packaging | 25.0 |
| ninja | 1.13.0 |
| setuptools | 80.10.2 |
| wheel | 0.46.3 |

The following CUDA extensions were available in the original environment:

```text
selective_scan_cuda_core
selective_scan_cuda_oflex
selective_scan_cuda
```

SCG-Mamba uses `forward_type="v2"`. Its backend wrapper selects implementations in the following order:

```text
selective_scan_cuda_core
    -> selective_scan_cuda_oflex
    -> selective_scan_cuda
```

Because `selective_scan_cuda_core` was available, it was the backend used for the reported experiments.

## Installation

Create and activate the environment:

```bash
conda create -n scg-mamba python=3.10.19 -y
conda activate scg-mamba
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Before compiling the reported selective-scan backend, verify that `kernels/selective_scan/setup.py` contains:

```python
MODES = ["core"]
```

Compile and install the extension using the active PyTorch/CUDA environment:

```bash
cd kernels/selective_scan
python -m pip install --no-build-isolation --no-deps .
cd ../..
```

Verify the installed backend. PyTorch must be imported first so that its shared libraries are loaded:

```bash
python -c "import torch; import selective_scan_cuda_core; print(selective_scan_cuda_core.__file__)"
```

Then run the Houston2013 training script:

```bash
python train_Houston2013_lite_m_clean_keep_gate2_basc_tail3_a005_seed42_aux060.py
```

## Optional packages

`mamba-ssm==2.2.2` and `transformers==4.43.3` were present in the collected environment, but they are not required when `selective_scan_cuda_core` is compiled and available. They are therefore omitted from the default requirements.

The collected environment also contained damaged pandas package metadata, which caused `import mamba_ssm` to fail through an optional Transformers/pandas version check. Pandas is not imported by the SCG-Mamba training pipeline and is not a required dependency.

## Reproducibility notes

- Do not commit compiled `.so` files, `build/`, or `*.egg-info/` directories.
- Compile `selective_scan_cuda_core` on the target Linux/PyTorch/CUDA environment.
- Record the GPU model and NVIDIA driver from the actual training session.
- Keep the fixed `TRAIN_GT.mat` and `TEST_GT.mat` split files unchanged.
- The Houston2013 patches are cropped online from the full HSI and LiDAR scenes; pre-generated patch files are not used.
