
# SCG-Mamba

<p align="center">
  <img src="assets/scg_mamba_framework.png"
       alt="Overall architecture of SCG-Mamba"
       width="95%">
</p>

<p align="center">
  <em>Figure 1. Overall architecture of the proposed SCG-Mamba.</em>
</p>
SCG-Mamba is a multimodal classification framework for hyperspectral imagery (HSI) and LiDAR data.


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



## Dependencies

| Category | Package/component | Version |
|---|---|---|
| PyTorch | PyTorch | 2.3.1+cu118 |
| PyTorch | TorchVision | 0.18.1+cu118 |
| PyTorch | TorchAudio | 2.3.1+cu118 |
| PyTorch | CUDA runtime | 11.8 |
| PyTorch | cuDNN | 8.7.0 |
| PyTorch | CXX11 ABI | False |
| PyTorch | Triton | 2.3.1 |
| Runtime | NumPy | 2.2.6 |
| Runtime | SciPy | 1.15.3 |
| Runtime | scikit-learn | 1.7.2 |
| Runtime | tqdm | 4.67.3 |
| Runtime | einops | 0.8.2 |
| Runtime | timm | 1.0.25 |
| Mamba | causal-conv1d | 1.4.0 |
| Mamba | mamba-ssm | 2.2.2 |
| Mamba | transformers | 4.43.3 |
| Build | packaging | 25.0 |
| Build | ninja | 1.13.0 |
| Build | setuptools | 80.10.2 |
| Build | wheel | 0.46.3 |

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

Create the Conda environment and install the dependencies:

```bash
conda create -n scg-mamba python=3.10.19 -y
conda activate scg-mamba

conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc=11.8.89

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torchaudio==2.3.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
python -m pip install causal-conv1d==1.4.0 --no-build-isolation
python -m pip install mamba-ssm==2.2.2 transformers==4.43.3 --no-build-isolation
```

> **Environment troubleshooting:** If you encounter network timeouts, CUDA version mismatches, or compilation failures, refer to this [Mamba environment setup guide](https://blog.csdn.net/qq_45645368/article/details/141031972).

The included `kernels/selective_scan/setup.py` is already configured with `MODES = ["core"]`. Compile the operator, verify the installation, and start training:

```bash
cd kernels/selective_scan
python -m pip install --no-build-isolation --no-deps .
cd ../..

python -c "import torch; import selective_scan_cuda_core; print(selective_scan_cuda_core.__file__)"

python train.py
```

The CUDA extension must be compiled in the same Python, PyTorch, and CUDA environment used to run SCG-Mamba.


## Dataset
The Houston2013 dataset is not stored directly in this repository. The dataset files used in our experiments can be downloaded from the following link:

- Dataset: Houston2013
- Download: [Baidu Netdisk](https://pan.baidu.com/s/1mp-SqFQYR6AuO1O6YdJN3A?pwd=3wf6)
- Extraction code: `3wf6`

After downloading, please modify the dataset file location in the code by yourself.

## Contact

If you have any questions regarding the code or experiments, please feel free to contact us at [vo1cee@163.com](mailto:vo1cee@163.com).
