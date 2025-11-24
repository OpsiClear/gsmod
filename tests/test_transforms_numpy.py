"""
Tests for NumPy implementation of quaternion utility functions.
"""

import numpy as np

from gsmod.transform import (
    axis_angle_to_quaternion,
    euler_to_quaternion,
    quaternion_multiply,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)

# ============================================================================
# Quaternion Operations Tests
# ============================================================================


def test_quaternion_multiply_numpy():
    """Test quaternion multiplication (NumPy)."""
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = np.array([0.7071, 0.0, 0.0, 0.7071])

    result = quaternion_multiply(q1, q2)

    assert isinstance(result, np.ndarray)
    # Result is [1, 4] due to batching, squeeze or compare with expanded q2
    np.testing.assert_allclose(result.squeeze(), q2, atol=1e-4)


def test_quaternion_to_rotation_matrix_numpy():
    """Test quaternion to rotation matrix conversion (NumPy)."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    R = quaternion_to_rotation_matrix(q)

    assert isinstance(R, np.ndarray)
    expected = np.eye(3)
    np.testing.assert_allclose(R, expected, atol=1e-6)


def test_rotation_matrix_to_quaternion_numpy():
    """Test rotation matrix to quaternion conversion (NumPy)."""
    R = np.eye(3)
    q = rotation_matrix_to_quaternion(R)

    assert isinstance(q, np.ndarray)
    expected = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(q, expected, atol=1e-6)


def test_quaternion_rotation_roundtrip_numpy():
    """Test quaternion <-> rotation matrix roundtrip (NumPy)."""
    q_original = np.random.randn(4)
    q_original = q_original / np.linalg.norm(q_original)

    R = quaternion_to_rotation_matrix(q_original)
    q_reconstructed = rotation_matrix_to_quaternion(R)

    # Quaternions q and -q represent same rotation
    assert np.allclose(q_reconstructed, q_original, atol=1e-5) or np.allclose(
        q_reconstructed, -q_original, atol=1e-5
    )


def test_axis_angle_to_quaternion_numpy():
    """Test axis-angle to quaternion conversion (NumPy)."""
    axis_angle = np.array([0.0, 0.0, 0.0])
    q = axis_angle_to_quaternion(axis_angle)

    assert isinstance(q, np.ndarray)
    expected = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(q, expected, atol=1e-6)


def test_axis_angle_90_degrees_numpy():
    """Test 90 degree rotation around Z axis (NumPy)."""
    axis_angle = np.array([0.0, 0.0, np.pi / 2])
    q = axis_angle_to_quaternion(axis_angle)

    expected = np.array([0.7071, 0.0, 0.0, 0.7071])
    np.testing.assert_allclose(q, expected, atol=1e-4)


def test_euler_to_quaternion_numpy():
    """Test Euler angles to quaternion conversion (NumPy)."""
    euler = np.array([0.0, 0.0, 0.0])
    q = euler_to_quaternion(euler)

    assert isinstance(q, np.ndarray)
    expected = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(q, expected, atol=1e-6)


def test_quaternion_to_euler_numpy():
    """Test quaternion to Euler angles conversion (NumPy)."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    euler = quaternion_to_euler(q)

    assert isinstance(euler, np.ndarray)
    expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(euler, expected, atol=1e-6)


def test_euler_quaternion_roundtrip_numpy():
    """Test Euler <-> quaternion roundtrip (NumPy)."""
    euler_original = np.array([0.5, 0.3, 0.7])

    q = euler_to_quaternion(euler_original)
    euler_reconstructed = quaternion_to_euler(q)

    np.testing.assert_allclose(euler_reconstructed, euler_original, atol=1e-5)


def test_zero_rotation_axis_angle_numpy():
    """Test that zero-length axis-angle gives identity quaternion (NumPy)."""
    axis_angle = np.array([0.0, 0.0, 0.0])
    q = axis_angle_to_quaternion(axis_angle)

    expected = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(q, expected, atol=1e-6)
