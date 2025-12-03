"""Triton-optimized kernels for SH color operations.

These fused kernels combine multiple operations into single kernel launches
for maximum GPU efficiency. Falls back to PyTorch if Triton is unavailable.
"""

from __future__ import annotations

import torch

# Try to import Triton, fall back gracefully if not available
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


if TRITON_AVAILABLE:

    @triton.jit
    def brightness_kernel(
        sh0_ptr,
        shN_ptr,
        output_sh0_ptr,
        output_shN_ptr,
        scale: tl.constexpr,
        N,
        K,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for brightness (scaling all SH bands)."""
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        # Scale sh0
        for c in range(3):
            idx = offsets * 3 + c
            val = tl.load(sh0_ptr + idx, mask=mask, other=0.0)
            tl.store(output_sh0_ptr + idx, val * scale, mask=mask)

        # Scale shN if present
        if K > 0:
            for k in range(K):
                for c in range(3):
                    idx = offsets * K * 3 + k * 3 + c
                    val = tl.load(shN_ptr + idx, mask=mask, other=0.0)
                    tl.store(output_shN_ptr + idx, val * scale, mask=mask)

    @triton.jit
    def contrast_kernel(
        sh0_ptr,
        output_sh0_ptr,
        factor: tl.constexpr,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for contrast adjustment (sh0 only, around 0.5 midpoint)."""
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        # Contrast: (val - 0.5) * factor + 0.5
        for c in range(3):
            idx = offsets * 3 + c
            val = tl.load(sh0_ptr + idx, mask=mask, other=0.0)
            out = (val - 0.5) * factor + 0.5
            tl.store(output_sh0_ptr + idx, out, mask=mask)

    @triton.jit
    def gamma_kernel(
        sh0_ptr,
        output_sh0_ptr,
        gamma: tl.constexpr,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for gamma correction (sh0 only)."""
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        # Gamma: val^gamma (clamp to avoid negative)
        for c in range(3):
            idx = offsets * 3 + c
            val = tl.load(sh0_ptr + idx, mask=mask, other=0.0)
            # Clamp to small positive value before power
            val_clamped = tl.maximum(val, 1e-8)
            out = tl.math.pow(val_clamped, gamma)
            tl.store(output_sh0_ptr + idx, out, mask=mask)

    @triton.jit
    def temperature_kernel(
        sh0_ptr,
        output_sh0_ptr,
        temp: tl.constexpr,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for temperature adjustment (sh0 only)."""
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        # Temperature: R *= (1 + temp*0.3), B *= (1 - temp*0.3)
        r_factor = 1.0 + temp * 0.3
        b_factor = 1.0 - temp * 0.3

        # Load RGB
        r = tl.load(sh0_ptr + offsets * 3 + 0, mask=mask, other=0.0)
        g = tl.load(sh0_ptr + offsets * 3 + 1, mask=mask, other=0.0)
        b = tl.load(sh0_ptr + offsets * 3 + 2, mask=mask, other=0.0)

        # Apply temperature
        tl.store(output_sh0_ptr + offsets * 3 + 0, r * r_factor, mask=mask)
        tl.store(output_sh0_ptr + offsets * 3 + 1, g, mask=mask)
        tl.store(output_sh0_ptr + offsets * 3 + 2, b * b_factor, mask=mask)

    @triton.jit
    def saturation_kernel(
        sh0_ptr,
        shN_ptr,
        output_sh0_ptr,
        output_shN_ptr,
        factor: tl.constexpr,
        N,
        K,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for saturation adjustment."""
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        # Load sh0
        r = tl.load(sh0_ptr + offsets * 3 + 0, mask=mask, other=0.0)
        g = tl.load(sh0_ptr + offsets * 3 + 1, mask=mask, other=0.0)
        b = tl.load(sh0_ptr + offsets * 3 + 2, mask=mask, other=0.0)

        # Compute luminance and apply saturation
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        r_out = lum + factor * (r - lum)
        g_out = lum + factor * (g - lum)
        b_out = lum + factor * (b - lum)

        # Store sh0
        tl.store(output_sh0_ptr + offsets * 3 + 0, r_out, mask=mask)
        tl.store(output_sh0_ptr + offsets * 3 + 1, g_out, mask=mask)
        tl.store(output_sh0_ptr + offsets * 3 + 2, b_out, mask=mask)

        # Apply to shN
        if K > 0:
            for k in range(K):
                r_sh = tl.load(shN_ptr + offsets * K * 3 + k * 3 + 0, mask=mask, other=0.0)
                g_sh = tl.load(shN_ptr + offsets * K * 3 + k * 3 + 1, mask=mask, other=0.0)
                b_sh = tl.load(shN_ptr + offsets * K * 3 + k * 3 + 2, mask=mask, other=0.0)

                lum_sh = 0.299 * r_sh + 0.587 * g_sh + 0.114 * b_sh
                r_sh_out = lum_sh + factor * (r_sh - lum_sh)
                g_sh_out = lum_sh + factor * (g_sh - lum_sh)
                b_sh_out = lum_sh + factor * (b_sh - lum_sh)

                tl.store(output_shN_ptr + offsets * K * 3 + k * 3 + 0, r_sh_out, mask=mask)
                tl.store(output_shN_ptr + offsets * K * 3 + k * 3 + 1, g_sh_out, mask=mask)
                tl.store(output_shN_ptr + offsets * K * 3 + k * 3 + 2, b_sh_out, mask=mask)


# Python wrappers
def triton_adjust_brightness(sh0: torch.Tensor, shN: torch.Tensor | None, factor: float):
    """Apply brightness using Triton kernel with fallback."""
    if not TRITON_AVAILABLE or not sh0.is_cuda:
        # Fallback: use PyTorch
        sh0_out = sh0 * factor
        shN_out = shN * factor if shN is not None else None
        return sh0_out, shN_out

    try:
        N = sh0.shape[0]
        K = shN.shape[1] if shN is not None and shN.numel() > 0 else 0

        # Allocate output
        sh0_out = torch.empty_like(sh0)
        shN_out = torch.empty_like(shN) if K > 0 else torch.empty(0, device=sh0.device)

        BLOCK_SIZE = 256
        grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        brightness_kernel[grid](
            sh0,
            shN if K > 0 else sh0,
            sh0_out,
            shN_out,
            factor,
            N,
            K,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return sh0_out, shN_out if K > 0 else None

    except Exception:
        # Fallback to PyTorch on any error
        sh0_out = sh0 * factor
        shN_out = shN * factor if shN is not None else None
        return sh0_out, shN_out


def triton_adjust_contrast(sh0: torch.Tensor, factor: float):
    """Apply contrast using Triton kernel with fallback."""
    if not TRITON_AVAILABLE or not sh0.is_cuda:
        # Fallback: use PyTorch
        return (sh0 - 0.5) * factor + 0.5

    try:
        N = sh0.shape[0]
        sh0_out = torch.empty_like(sh0)

        BLOCK_SIZE = 256
        grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        contrast_kernel[grid](
            sh0,
            sh0_out,
            factor,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return sh0_out

    except Exception:
        return (sh0 - 0.5) * factor + 0.5


def triton_adjust_gamma(sh0: torch.Tensor, gamma: float):
    """Apply gamma correction using Triton kernel with fallback."""
    if not TRITON_AVAILABLE or not sh0.is_cuda:
        # Fallback: use PyTorch
        return torch.pow(torch.clamp(sh0, min=1e-8), gamma)

    try:
        N = sh0.shape[0]
        sh0_out = torch.empty_like(sh0)

        BLOCK_SIZE = 256
        grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        gamma_kernel[grid](
            sh0,
            sh0_out,
            gamma,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return sh0_out

    except Exception:
        return torch.pow(torch.clamp(sh0, min=1e-8), gamma)


def triton_adjust_temperature(sh0: torch.Tensor, temp: float):
    """Apply temperature adjustment using Triton kernel with fallback."""
    if not TRITON_AVAILABLE or not sh0.is_cuda:
        # Fallback: use PyTorch
        r_factor = 1.0 + temp * 0.3
        b_factor = 1.0 - temp * 0.3
        sh0_out = sh0.clone()
        sh0_out[:, 0] *= r_factor
        sh0_out[:, 2] *= b_factor
        return sh0_out

    try:
        N = sh0.shape[0]
        sh0_out = torch.empty_like(sh0)

        BLOCK_SIZE = 256
        grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        temperature_kernel[grid](
            sh0,
            sh0_out,
            temp,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return sh0_out

    except Exception:
        r_factor = 1.0 + temp * 0.3
        b_factor = 1.0 - temp * 0.3
        sh0_out = sh0.clone()
        sh0_out[:, 0] *= r_factor
        sh0_out[:, 2] *= b_factor
        return sh0_out


def triton_adjust_saturation(sh0: torch.Tensor, shN: torch.Tensor | None, factor: float):
    """Apply saturation using Triton kernel with fallback."""
    if not TRITON_AVAILABLE or not sh0.is_cuda:
        # Fallback: use PyTorch saturation matrix
        from gsmod.torch.sh_utils import apply_matrix_to_sh, build_saturation_matrix

        M = build_saturation_matrix(factor)
        return apply_matrix_to_sh(sh0, shN, M)

    try:
        N = sh0.shape[0]
        K = shN.shape[1] if shN is not None and shN.numel() > 0 else 0

        # Allocate output
        sh0_out = torch.empty_like(sh0)
        shN_out = torch.empty_like(shN) if K > 0 else torch.empty(0, device=sh0.device)

        BLOCK_SIZE = 256
        grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        saturation_kernel[grid](
            sh0,
            shN if K > 0 else sh0,
            sh0_out,
            shN_out,
            factor,
            N,
            K,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return sh0_out, shN_out if K > 0 else None

    except Exception:
        # Fallback to PyTorch
        from gsmod.torch.sh_utils import apply_matrix_to_sh, build_saturation_matrix

        M = build_saturation_matrix(factor)
        return apply_matrix_to_sh(sh0, shN, M)


def is_triton_available() -> bool:
    """Check if Triton is available and working."""
    return TRITON_AVAILABLE
