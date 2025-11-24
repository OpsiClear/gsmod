import numpy as np
import torch
from gsply import GSData

from gsmod import Color
from gsmod.torch import ColorGPU, GSTensorPro


def create_test_data(n_gaussians=1000, seed=42):
    np.random.seed(seed)
    data = GSData(
        means=np.random.randn(n_gaussians, 3).astype(np.float32) * 2,
        scales=np.random.rand(n_gaussians, 3).astype(np.float32) * 0.5 + 0.1,
        quats=np.random.randn(n_gaussians, 4).astype(np.float32),
        opacities=np.random.rand(n_gaussians).astype(np.float32),
        sh0=np.random.rand(n_gaussians, 3).astype(np.float32),
        shN=None,
    )
    norms = np.linalg.norm(data.quats, axis=1, keepdims=True)
    data.quats /= norms
    return data


def check_diff():
    data = create_test_data()
    factor = 1.2

    # CPU
    cpu_result = Color().brightness(factor)(data.copy(), inplace=True)

    # GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gstensor = GSTensorPro.from_gsdata(data.copy(), device=device)
    gpu_result = ColorGPU().brightness(factor)(gstensor, inplace=True)
    gpu_data = gpu_result.to_gsdata()

    diff = np.abs(cpu_result.sh0 - gpu_data.sh0)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")

    # Check if it passes with 1e-2
    if max_diff < 1e-2:
        print("Passes with 1e-2")
    else:
        print("Fails with 1e-2")


if __name__ == "__main__":
    check_diff()
