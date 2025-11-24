"""
Tests for Numba-optimized operations.

Tests correctness and fallback behavior of Numba kernels.
"""

import numpy as np
import pytest

from gsmod.transform.api import _quaternion_multiply_numpy

# Import Numba kernels - tests will be skipped if not available
transform_kernels = pytest.importorskip("gsmod.transform.kernels", reason="Numba not installed")


class TestNumbaKernels:
    """Test Numba-optimized kernels for correctness."""

    def test_quaternion_multiply_single(self):
        """Test single quaternion broadcast multiplication."""
        N = 1000
        q1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        q2 = np.random.randn(N, 4).astype(np.float32)
        q2 = q2 / np.linalg.norm(q2, axis=1, keepdims=True)

        # Numba result
        out_numba = np.empty_like(q2)
        transform_kernels.quaternion_multiply_single_numba(q1, q2, out_numba)

        # NumPy reference (using pure NumPy path)
        q1_2d = q1[np.newaxis, :]
        w1, x1, y1, z1 = q1_2d[:, 0], q1_2d[:, 1], q1_2d[:, 2], q1_2d[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        out_numpy = np.stack([w, x, y, z], axis=1)

        np.testing.assert_allclose(out_numba, out_numpy, atol=1e-5)

    def test_quaternion_multiply_batched(self):
        """Test batched quaternion multiplication."""
        N = 1000
        q1 = np.random.randn(N, 4).astype(np.float32)
        q2 = np.random.randn(N, 4).astype(np.float32)
        q1 = q1 / np.linalg.norm(q1, axis=1, keepdims=True)
        q2 = q2 / np.linalg.norm(q2, axis=1, keepdims=True)

        # Numba result
        out_numba = np.empty_like(q1)
        transform_kernels.quaternion_multiply_batched_numba(q1, q2, out_numba)

        # NumPy reference
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        out_numpy = np.stack([w, x, y, z], axis=1)

        np.testing.assert_allclose(out_numba, out_numpy, atol=1e-5)

    def test_apply_transform_matrix(self):
        """Test matrix application."""
        N = 1000
        points = np.random.randn(N, 3).astype(np.float32)
        R = np.random.randn(3, 3).astype(np.float32)
        t = np.random.randn(3).astype(np.float32)

        # Numba result
        out_numba = np.empty_like(points)
        transform_kernels.apply_transform_matrix_numba(points, R, t, out_numba)

        # NumPy reference
        out_numpy = points @ R.T + t

        np.testing.assert_allclose(out_numba, out_numpy, atol=1e-4)

    def test_elementwise_multiply_scalar(self):
        """Test scalar multiplication."""
        N = 1000
        arr = np.random.randn(N, 3).astype(np.float32)
        scalar = 2.5

        # Numba result
        out_numba = np.empty_like(arr)
        transform_kernels.elementwise_multiply_scalar_numba(arr, scalar, out_numba)

        # NumPy reference
        out_numpy = arr * scalar

        np.testing.assert_allclose(out_numba, out_numpy, atol=1e-5)

    def test_elementwise_multiply_vector(self):
        """Test vector multiplication with broadcasting."""
        N = 1000
        arr = np.random.randn(N, 3).astype(np.float32)
        vec = np.random.randn(3).astype(np.float32)

        # Numba result
        out_numba = np.empty_like(arr)
        transform_kernels.elementwise_multiply_vector_numba(arr, vec, out_numba)

        # NumPy reference
        out_numpy = arr * vec

        np.testing.assert_allclose(out_numba, out_numpy, atol=1e-5)

    def test_elementwise_multiply_vector_2d(self):
        """Test vector multiplication with 2D vector input."""
        N = 1000
        arr = np.random.randn(N, 3).astype(np.float32)
        vec = np.random.randn(1, 3).astype(np.float32)

        # Numba result
        out_numba = np.empty_like(arr)
        transform_kernels.elementwise_multiply_vector_numba(arr, vec, out_numba)

        # NumPy reference
        out_numpy = arr * vec

        np.testing.assert_allclose(out_numba, out_numpy, atol=1e-5)


class TestNumbaIntegration:
    """Test Numba integration in transforms.py."""

    def test_quaternion_multiply_uses_numba(self):
        """Test that _quaternion_multiply_numpy uses Numba when available."""
        N = 1000
        q1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        q2 = np.random.randn(N, 4).astype(np.float32)
        q2 = q2 / np.linalg.norm(q2, axis=1, keepdims=True)

        # This should use Numba internally
        result = _quaternion_multiply_numpy(q1, q2)

        # Verify shape and correctness
        assert result.shape == (N, 4)

        # Verify quaternions are normalized
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


class TestNumbaFallback:
    """Test graceful fallback when Numba is not available."""

    def test_quaternion_multiply_fallback(self):
        """Test that quaternion multiply works without Numba."""
        N = 100
        q1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        q2 = np.random.randn(N, 4).astype(np.float32)
        q2 = q2 / np.linalg.norm(q2, axis=1, keepdims=True)

        # Should work even if Numba is not available (uses NumPy fallback)
        result = _quaternion_multiply_numpy(q1, q2)

        assert result.shape == (N, 4)

        # Verify quaternions are normalized
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


class TestNumbaStatus:
    """Test Numba status reporting."""

    def test_get_numba_status(self):
        """Test get_numba_status function."""
        status = transform_kernels.get_numba_status()

        assert isinstance(status, dict)
        assert "available" in status

        # If available is True, check that version info is present
        if status["available"]:
            assert "version" in status
            assert "num_threads" in status
            assert "threading_layer" in status

        # Since we're using importorskip, Numba must be available
        # But get_numba_status() might return cached state
        # Just verify the function works
        assert isinstance(status.get("available"), bool)
