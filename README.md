[ENVIRONMENT_SCG_MAMBA.md](https://github.com/user-attachments/files/30686698/ENVIRONMENT_SCG_MAMBA.md)
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

## Visual Mamba selective-scan backend

The standard `mamba-ssm` package primarily provides the one-dimensional sequence Mamba implementation. SCG-Mamba is based on the two-dimensional visual selective-scan mechanism used by Visual Mamba; therefore, installing `mamba-ssm` alone is not sufficient to reproduce the backend used in this project.

SCG-Mamba uses the VMamba-style `selective_scan_cuda_core` extension. The CUDA source code required for compilation is included in:

```text
kernels/selective_scan/
```

The selective-scan implementation included in this repository is derived from the original [MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba) repository. Please refer to the original repository for the upstream implementation and additional information.

The reported experiments use:

```text
forward_type="v2"
backend=selective_scan_cuda_core
```

The precompiled `.so` file is not included because CUDA extensions depend on the target operating system, Python, PyTorch, CUDA Toolkit, and GPU architecture. Users should compile the extension locally in their own environment.
## Installation

Create and activate the Conda environment:

```bash
conda create -n scg-mamba python=3.10.19 -y
conda activate scg-mamba
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **Environment troubleshooting:** Configuring a Mamba-related environment may involve issues such as network timeouts, incompatible PyTorch and CUDA versions, missing CUDA compiler components, or CUDA extension compilation failures. For general troubleshooting and offline package installation strategies, users may refer to this [Mamba environment setup guide](https://blog.csdn.net/qq_45645368/article/details/141031972). This guide is provided only as an additional reference; the package versions specified in `requirements.txt` and `ENVIRONMENT_SCG_MAMBA.md` should take precedence for reproducing SCG-Mamba.

The included `kernels/selective_scan/setup.py` is already configured with:

```python
MODES = ["core"]
```

Therefore, no manual modification of `setup.py` is required.

Compile and install the Visual Mamba selective-scan operator using the source code included in this repository:

```bash
cd kernels/selective_scan
python -m pip install --no-build-isolation --no-deps .
cd ../..
```

The extension must be compiled in the same active Python, PyTorch, and CUDA environment that will be used to run SCG-Mamba. Do not directly reuse a precompiled `.so` file from another environment.

Verify the installed backend. PyTorch must be imported first so that its shared libraries are loaded:

```bash
python -c "import torch; import selective_scan_cuda_core; print(selective_scan_cuda_core.__file__)"
```

If the command prints the path of a `selective_scan_cuda_core` `.so` file without reporting an error, the operator has been installed successfully.

Then run the Houston2013 training script:

```bash
python train_Houston2013_lite_m_clean_keep_gate2_basc_tail3_a005_seed42_aux060.py
```


## Dataset
The Houston2013 dataset is not stored directly in this repository. The dataset files used in our experiments can be downloaded from the following link:

- Dataset: Houston2013
- Download: [Baidu Netdisk](https://pan.baidu.com/s/1mp-SqFQYR6AuO1O6YdJN3A?pwd=3wf6)
- Extraction code: `3wf6`

After downloading, extract the dataset and organize the files according to the directory structure shown below. The dataset is provided for academic research purposes only. Please comply with its applicable terms of use.


