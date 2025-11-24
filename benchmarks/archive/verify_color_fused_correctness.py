"""
Verify correctness of fused color Phase 2 kernel.
"""

import torch

from gsmod import ColorLUT

print("=" * 80)
print("COLOR FUSED KERNEL CORRECTNESS VERIFICATION")
print("=" * 80)

N = 100_000
colors = torch.rand(N, 3, device="cpu")

# Create two ColorLUT instances
lut_standard = ColorLUT(device="cpu", lut_size=1024)
lut_fused = ColorLUT(device="cpu", lut_size=1024)

# Temporarily disable Numba for standard path
import gsmod.color as color_module

original_numba_available = color_module.NUMBA_AVAILABLE
color_module.NUMBA_AVAILABLE = False

print("\n[Test 1: Fused vs Standard Path]")

test_params = [
    ("Default (all 1.0)", 1.0, 1.0, 1.0),
    ("Saturation only", 1.3, 1.0, 1.0),
    ("Shadows only", 1.0, 1.1, 1.0),
    ("Highlights only", 1.0, 1.0, 0.9),
    ("All Phase 2 ops", 1.3, 1.1, 0.9),
    ("Extreme saturation", 0.0, 1.0, 1.0),  # Grayscale
    ("Extreme shadows", 1.0, 2.0, 1.0),
    ("Extreme highlights", 1.0, 1.0, 0.5),
]

all_pass = True

for name, sat, shad, high in test_params:
    # Standard path
    result_standard = lut_standard.apply(
        colors,
        saturation=sat,
        shadows=shad,
        highlights=high,
        temperature=0.5,
        brightness=1.0,
        contrast=1.0,
        gamma=1.0,
    )

    # Re-enable Numba for fused path
    color_module.NUMBA_AVAILABLE = True
    result_fused = lut_fused.apply(
        colors,
        saturation=sat,
        shadows=shad,
        highlights=high,
        temperature=0.5,
        brightness=1.0,
        contrast=1.0,
        gamma=1.0,
    )
    color_module.NUMBA_AVAILABLE = False

    # Compare
    diff = torch.abs(result_standard - result_fused)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    tolerance = 1e-5
    passed = max_diff < tolerance

    status = "[PASS]" if passed else "[FAIL]"
    print(f"\n{name}:")
    print(f"  Max diff:  {max_diff:.2e} {status}")
    print(f"  Mean diff: {mean_diff:.2e}")

    if not passed:
        all_pass = False
        # Show where differences occur
        print(f"  Large diffs at indices: {torch.where(diff > tolerance)[0][:5].tolist()}")

# Restore original state
color_module.NUMBA_AVAILABLE = original_numba_available

print("\n" + "=" * 80)
if all_pass:
    print("[OK] ALL TESTS PASS - Fused kernel is correct!")
    print("[OK] Differences within float32 precision (< 1e-5)")
else:
    print("[FAIL] Some tests failed - investigate differences")
print("=" * 80)
