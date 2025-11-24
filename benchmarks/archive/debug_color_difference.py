"""
Debug the difference in color processing.
"""

import torch

from gsmod import ColorLUT

# Single pixel test
colors = torch.tensor([[0.3, 0.6, 0.4]], device="cpu")

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

print("Input:", colors[0])
print("Standard result:", result_standard[0])

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

print("Fused result:", result_fused[0])
print("Difference:", (result_standard - result_fused)[0])

# Manual calculation
r, g, b = 0.3, 0.6, 0.4

# After Phase 1 (LUT), colors are unchanged with defaults
print("\n[Manual Calculation]")
print(f"After Phase 1: R={r:.4f}, G={g:.4f}, B={b:.4f}")

# Luminance before saturation
lum_before = 0.299 * r + 0.587 * g + 0.114 * b
print(f"Luminance (before saturation): {lum_before:.4f}")

# Apply saturation
saturation = 1.3
r_sat = lum_before + saturation * (r - lum_before)
g_sat = lum_before + saturation * (g - lum_before)
b_sat = lum_before + saturation * (b - lum_before)
print(f"After saturation: R={r_sat:.4f}, G={g_sat:.4f}, B={b_sat:.4f}")

# Luminance after saturation (for shadows/highlights)
lum_after = 0.299 * r_sat + 0.587 * g_sat + 0.114 * b_sat
print(f"Luminance (after saturation): {lum_after:.4f}")

# Apply shadows/highlights based on luminance AFTER saturation
if lum_after < 0.5:
    print("Shadow region - apply shadows=1.1")
    shadows = 1.1
    factor = shadows - 1.0
    r_final = r_sat + r_sat * factor
    g_final = g_sat + g_sat * factor
    b_final = b_sat + b_sat * factor
else:
    print("Highlight region - apply highlights=0.9")
    highlights = 0.9
    factor = highlights - 1.0
    r_final = r_sat + r_sat * factor
    g_final = g_sat + g_sat * factor
    b_final = b_sat + b_sat * factor

print(f"Final (manual): R={r_final:.4f}, G={g_final:.4f}, B={b_final:.4f}")

color_module.NUMBA_AVAILABLE = original_numba_available
