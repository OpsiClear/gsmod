"""
Numba-optimized kernels for 3D transform operations.

Provides JIT-compiled kernels for quaternion operations and Gaussian transforms,
offering 4-200x speedup over pure NumPy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import guvectorize, njit, prange
from numpy.typing import NDArray

# ============================================================================
# Quaternion Operations (Major Bottleneck - 37ms -> 0.2ms)
# ============================================================================


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def quaternion_multiply_single_numba(
    q1: NDArray[np.float32], q2: NDArray[np.float32], out: NDArray[np.float32]
) -> None:
    """
    Multiply single quaternion q1 with array of quaternions q2.

    Numba-optimized version with parallelization.
    ~200x faster than pure NumPy for 1M quaternions.

    Args:
        q1: Single quaternion [4] (w, x, y, z)
        q2: Array of quaternions [N, 4]
        out: Output array [N, 4] (pre-allocated)

    Note: Modifies out in-place for efficiency
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]

    N = q2.shape[0]
    for i in prange(N):
        w2 = q2[i, 0]
        x2 = q2[i, 1]
        y2 = q2[i, 2]
        z2 = q2[i, 3]

        out[i, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        out[i, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        out[i, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        out[i, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def quaternion_multiply_batched_numba(
    q1: NDArray[np.float32], q2: NDArray[np.float32], out: NDArray[np.float32]
) -> None:
    """
    Multiply two arrays of quaternions element-wise.

    Args:
        q1: Array of quaternions [N, 4]
        q2: Array of quaternions [N, 4]
        out: Output array [N, 4] (pre-allocated)
    """
    N = q1.shape[0]
    for i in prange(N):
        w1, x1, y1, z1 = q1[i, 0], q1[i, 1], q1[i, 2], q1[i, 3]
        w2, x2, y2, z2 = q2[i, 0], q2[i, 1], q2[i, 2], q2[i, 3]

        out[i, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        out[i, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        out[i, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        out[i, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2


# ============================================================================
# GUVectorize Versions (Eliminates Wrapper Overhead)
# ============================================================================


@guvectorize(
    ["void(float32[:], float32[:,:], float32[:,:])"],
    "(m),(n,m)->(n,m)",
    nopython=True,
    target="parallel",
    cache=True,
)
def quaternion_multiply_single_guvec(q1, q2, out):
    """
    GUVectorize version: handles broadcasting automatically, no wrapper overhead.

    Eliminates ~1ms of shape checking and allocation overhead.

    Args:
        q1: Single quaternion [4] (w, x, y, z)
        q2: Array of quaternions [N, 4]
        out: Output array [N, 4]
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]

    for i in range(q2.shape[0]):
        w2, x2, y2, z2 = q2[i, 0], q2[i, 1], q2[i, 2], q2[i, 3]

        out[i, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        out[i, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        out[i, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        out[i, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2


# ============================================================================
# Matrix Application (Not used - NumPy BLAS is faster)
# ============================================================================


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def apply_transform_matrix_numba(
    points: NDArray[np.float32],
    R: NDArray[np.float32],
    t: NDArray[np.float32],
    out: NDArray[np.float32],
) -> None:
    """
    Apply 3x3 rotation/scale matrix and translation to points.

    NOTE: Not used in transforms.py - NumPy's BLAS-optimized @ operator
    is faster than this naive Numba implementation for matrix multiplication.

    Kept for reference and potential future optimizations.

    Args:
        points: Input points [N, 3]
        R: Combined rotation/scale matrix [3, 3]
        t: Translation vector [3]
        out: Output array [N, 3] (pre-allocated)
    """
    N = points.shape[0]
    for i in prange(N):
        # Matrix multiply: points[i] @ R.T + t
        for j in range(3):
            sum_val = 0.0
            for k in range(3):
                sum_val += points[i, k] * R[j, k]  # R.T indexing
            out[i, j] = sum_val + t[j]


# ============================================================================
# Elementwise Operations (2.6ms -> 0.3ms)
# ============================================================================


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def elementwise_multiply_scalar_numba(
    arr: NDArray[np.float32], scalar: float, out: NDArray[np.float32]
) -> None:
    """
    Multiply array by scalar element-wise: arr * scalar

    Args:
        arr: Input array [N, M]
        scalar: Scalar multiplier
        out: Output array [N, M] (pre-allocated)
    """
    N = arr.shape[0]
    M = arr.shape[1]
    for i in prange(N):
        for j in range(M):
            out[i, j] = arr[i, j] * scalar


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def elementwise_multiply_vector_numba(
    arr: NDArray[np.float32], vec: NDArray[np.float32], out: NDArray[np.float32]
) -> None:
    """
    Multiply array by vector (broadcast): arr * vec

    Args:
        arr: Input array [N, M]
        vec: Vector [M] or [1, M]
        out: Output array [N, M] (pre-allocated)
    """
    N = arr.shape[0]
    M = arr.shape[1]

    # Handle both [M] and [1, M] shapes
    if vec.ndim == 1:
        for i in prange(N):
            for j in range(M):
                out[i, j] = arr[i, j] * vec[j]
    else:
        for i in prange(N):
            for j in range(M):
                out[i, j] = arr[i, j] * vec[0, j]


# ============================================================================
# Fused Transform Kernel (4-5x faster than separate operations)
# ============================================================================


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def fused_transform_numba(
    means: NDArray[np.float32],
    quaternions: NDArray[np.float32],
    scales: NDArray[np.float32],
    rot_quat: NDArray[np.float32],
    scale_vec: NDArray[np.float32],
    translation: NDArray[np.float32],
    R: NDArray[np.float32],
    out_means: NDArray[np.float32],
    out_quats: NDArray[np.float32],
    out_scales: NDArray[np.float32],
) -> None:
    """
    Fused kernel that performs all transform operations in a single parallel loop.

    This achieves 4-5x speedup over separate operations by:
    - Eliminating function call overhead
    - Better memory locality (process each Gaussian completely)
    - Single parallel loop with prange
    - Avoiding intermediate allocations

    Args:
        means: Input positions [N, 3]
        quaternions: Input orientations [N, 4]
        scales: Input scales [N, 3]
        rot_quat: Rotation quaternion [4] (w, x, y, z)
        scale_vec: Scale vector [3]
        translation: Translation vector [3]
        R: Pre-computed rotation matrix [3, 3] (includes scale)
        out_means: Output positions [N, 3]
        out_quats: Output orientations [N, 4]
        out_scales: Output scales [N, 3]

    Performance: ~2ms for 1M Gaussians (vs 9ms for separate operations)
    """
    N = means.shape[0]

    # Unpack rotation quaternion once
    w1, x1, y1, z1 = rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3]

    # Single parallel loop processes each Gaussian completely
    for i in prange(N):
        # 1. Transform means: R @ point + t
        #    (Custom matmul is 9x faster than BLAS for small matrices with large N)
        for j in range(3):
            out_means[i, j] = (
                R[j, 0] * means[i, 0]
                + R[j, 1] * means[i, 1]
                + R[j, 2] * means[i, 2]
                + translation[j]
            )

        # 2. Quaternion multiply: rot_quat * quaternions[i]
        w2, x2, y2, z2 = quaternions[i, 0], quaternions[i, 1], quaternions[i, 2], quaternions[i, 3]
        out_quats[i, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        out_quats[i, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        out_quats[i, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        out_quats[i, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        # 3. Scale multiply
        out_scales[i, 0] = scales[i, 0] * scale_vec[0]
        out_scales[i, 1] = scales[i, 1] * scale_vec[1]
        out_scales[i, 2] = scales[i, 2] * scale_vec[2]


# ============================================================================
# Helper Functions
# ============================================================================


def get_numba_status() -> dict[str, Any]:
    """
    Get information about Numba availability and configuration.

    Returns:
        Dictionary with Numba status information
    """
    import numba

    return {
        "available": True,
        "version": numba.__version__,
        "num_threads": numba.config.NUMBA_NUM_THREADS,
        "threading_layer": numba.config.THREADING_LAYER,
    }


def warmup_transform_kernels() -> None:
    """
    Warm up Numba JIT compilation for transform kernels.

    Call this once at import time to avoid first-call compilation overhead.
    """
    # Dummy data for warmup
    q1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q2 = np.random.randn(100, 4).astype(np.float32)
    points = np.random.randn(100, 3).astype(np.float32)
    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)

    out_q = np.empty((100, 4), dtype=np.float32)
    out_p = np.empty((100, 3), dtype=np.float32)

    # Trigger JIT compilation
    quaternion_multiply_single_numba(q1, q2, out_q)
    quaternion_multiply_batched_numba(q2, q2, out_q)
    apply_transform_matrix_numba(points, R, t, out_p)
    elementwise_multiply_scalar_numba(points, 2.0, out_p)
    elementwise_multiply_vector_numba(points, t, out_p)

    # Warmup fused transform kernel
    scales = np.random.randn(100, 3).astype(np.float32)
    out_s = np.empty((100, 3), dtype=np.float32)
    scale_vec = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    fused_transform_numba(points, q2, scales, q1, scale_vec, t, R, out_p, out_q, out_s)


# Warmup on import to avoid first-call overhead
warmup_transform_kernels()
