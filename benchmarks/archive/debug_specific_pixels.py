"""
Debug specific pixels that show large differences.
"""

import numpy as np
import torch

from gsmod import ColorLUT

N = 100_000
np.random.seed(42)
torch.manual_seed(42)
colors = torch.rand(N, 3, device="cpu")

lut = ColorLUT(device="cpu", lut_size=1024)

# Disable Numba temporarily
import gsmod.color as color_module

original_numba_available = color_module.NUMBA_AVAILABLE
color_module.NUMBA_AVAILABLE = False

result_standard = lut.apply(
    colors,
    saturation=1.3,
    shadows=1.1,
    highlights=0.9,
    temperature=0.5,
    brightness=1.0,
    contrast=1.0,
    gamma=1.0,
)

# Enable Numba
color_module.NUMBA_AVAILABLE = True
result_fused = lut.apply(
    colors,
    saturation=1.3,
    shadows=1.1,
    highlights=0.9,
    temperature=0.5,
    brightness=1.0,
    contrast=1.0,
    gamma=1.0,
)

# Find pixels with large differences
diff = torch.abs(result_standard - result_fused)
large_diff_mask = diff.max(dim=1)[0] > 1e-3  # More than 0.001 difference
problem_indices = torch.where(large_diff_mask)[0]

print(f"Found {len(problem_indices)} pixels with large differences (> 0.001)")
print(f"Problematic indices: {problem_indices[:10].tolist()}")

if len(problem_indices) > 0:
    for idx in problem_indices[:5]:
        print(f"\n[Pixel {idx}]")
        print(f"  Input:    {colors[idx]}")
        print(f"  Standard: {result_standard[idx]}")
        print(f"  Fused:    {result_fused[idx]}")
        print(f"  Diff:     {(result_standard - result_fused)[idx]}")
        print(f"  Max diff: {diff[idx].max():.4f}")

        # Calculate luminance
        r, g, b = colors[idx][0].item(), colors[idx][1].item(), colors[idx][2].item()
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        print(f"  Luminance: {lum:.4f} ({'Shadow' if lum < 0.5 else 'Highlight'} region)")

color_module.NUMBA_AVAILABLE = original_numba_available
