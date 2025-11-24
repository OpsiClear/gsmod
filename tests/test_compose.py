"""
Tests for scene composition utilities.
"""

import numpy as np
import pytest
from gsply import GSData

from gsmod import Transform, compose_with_transforms, concatenate, merge_scenes, split_by_region
from gsmod.compose import deduplicate

# Check if gsply has concatenate/add support
_test_data = GSData(
    means=np.zeros((1, 3), dtype=np.float32),
    scales=np.zeros((1, 3), dtype=np.float32),
    quats=np.array([[1, 0, 0, 0]], dtype=np.float32),
    opacities=np.zeros(1, dtype=np.float32),
    sh0=np.zeros((1, 3), dtype=np.float32),
    shN=None,
)
_gsdata_has_concatenate = hasattr(GSData, "concatenate")
_gsdata_has_add = hasattr(_test_data, "add")

skip_no_concat_support = pytest.mark.skipif(
    not _gsdata_has_concatenate or not _gsdata_has_add,
    reason="gsply version does not support GSData.concatenate() or .add()",
)


def create_test_scene(n_gaussians=100, offset=(0.0, 0.0, 0.0), seed=None):
    """Create a test GSData scene with specified offset."""
    if seed is not None:
        np.random.seed(seed)
    means = np.random.randn(n_gaussians, 3).astype(np.float32)
    means += np.array(offset, dtype=np.float32)

    quats = np.random.randn(n_gaussians, 4).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)  # Normalize

    scales = np.abs(np.random.randn(n_gaussians, 3).astype(np.float32)) * 0.1

    opacities = np.random.rand(n_gaussians).astype(np.float32)

    sh0 = np.random.rand(n_gaussians, 3).astype(np.float32)

    return GSData(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        sh0=sh0,
        shN=None,
    )


@skip_no_concat_support
def test_concatenate_basic():
    """Test basic concatenation of multiple scenes."""
    scene1 = create_test_scene(100, seed=1)
    scene2 = create_test_scene(150, seed=2)
    scene3 = create_test_scene(200, seed=3)

    result = concatenate([scene1, scene2, scene3])

    assert len(result) == 100 + 150 + 200
    assert result.means.shape == (450, 3)
    assert result.quats.shape == (450, 4)
    assert result.scales.shape == (450, 3)


def test_concatenate_single_scene():
    """Test concatenation with single scene (should return copy)."""
    scene = create_test_scene(100)
    result = concatenate([scene])

    assert len(result) == len(scene)
    assert result is not scene  # Should be a copy
    np.testing.assert_array_equal(result.means, scene.means)


@skip_no_concat_support
def test_concatenate_two_scenes():
    """Test concatenation of exactly two scenes (uses .add())."""
    scene1 = create_test_scene(100, seed=1)
    scene2 = create_test_scene(150, seed=2)

    result = concatenate([scene1, scene2])

    assert len(result) == 250
    # First 100 should be from scene1
    np.testing.assert_array_equal(result.means[:100], scene1.means)
    # Next 150 should be from scene2
    np.testing.assert_array_equal(result.means[100:], scene2.means)


def test_concatenate_empty():
    """Test concatenation with empty list raises error."""
    with pytest.raises(ValueError, match="Cannot concatenate empty list"):
        concatenate([])


@skip_no_concat_support
def test_compose_with_transforms():
    """Test composition with transformations applied."""
    scene1 = create_test_scene(100, offset=(0, 0, 0), seed=1)
    scene2 = create_test_scene(100, offset=(0, 0, 0), seed=2)

    # Transform scene2 to the right
    t1 = Transform()  # Identity
    t2 = Transform().translate([5.0, 0.0, 0.0])

    result = compose_with_transforms([scene1, scene2], [t1, t2], inplace=False)

    assert len(result) == 200

    # First 100 should be at original position
    np.testing.assert_array_almost_equal(result.means[:100], scene1.means, decimal=5)

    # Second 100 should be translated
    expected = scene2.means + np.array([5.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_array_almost_equal(result.means[100:], expected, decimal=5)


def test_compose_with_transforms_mismatch():
    """Test error when scenes and transforms counts don't match."""
    scene1 = create_test_scene(100)
    scene2 = create_test_scene(100)

    t1 = Transform()

    with pytest.raises(ValueError, match="Number of scenes.*must match"):
        compose_with_transforms([scene1, scene2], [t1])


@skip_no_concat_support
def test_compose_with_identity_transform():
    """Test composition where all transforms are identity (fast path)."""
    scene1 = create_test_scene(100, seed=1)
    scene2 = create_test_scene(100, seed=2)

    t1 = Transform()  # Identity
    t2 = Transform()  # Identity

    result = compose_with_transforms([scene1, scene2], [t1, t2], inplace=False)

    assert len(result) == 200


def test_deduplicate_basic():
    """Test deduplication of nearby Gaussians."""
    # Create scene with duplicates
    means = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # Duplicate
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Duplicate
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    n = len(means)
    scene = GSData(
        means=means,
        quats=np.tile([0, 0, 0, 1], (n, 1)).astype(np.float32),
        scales=np.ones((n, 3), dtype=np.float32) * 0.1,
        opacities=np.ones(n, dtype=np.float32),
        sh0=np.ones((n, 3), dtype=np.float32) * 0.5,
        shN=None,
    )

    result = deduplicate(scene, position_threshold=1e-6, method="first")

    # Should keep only 3 unique positions
    assert len(result) == 3


def test_deduplicate_threshold():
    """Test deduplication with distance threshold."""
    # Create scene with near-duplicates
    means = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.001, 0.001, 0.001],  # Within threshold
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    n = len(means)
    scene = GSData(
        means=means,
        quats=np.tile([0, 0, 0, 1], (n, 1)).astype(np.float32),
        scales=np.ones((n, 3), dtype=np.float32) * 0.1,
        opacities=np.ones(n, dtype=np.float32),
        sh0=np.ones((n, 3), dtype=np.float32) * 0.5,
        shN=None,
    )

    # With small threshold, should remove near-duplicate
    result = deduplicate(scene, position_threshold=0.01, method="first")
    assert len(result) == 2

    # With large threshold, should keep all
    result = deduplicate(scene, position_threshold=0.0001, method="first")
    assert len(result) == 3


@skip_no_concat_support
def test_merge_scenes_basic():
    """Test high-level merge_scenes API."""
    scene1 = create_test_scene(100, seed=1)
    scene2 = create_test_scene(100, seed=2)
    scene3 = create_test_scene(100, seed=3)

    result = merge_scenes([scene1, scene2, scene3])

    assert len(result) == 300


@skip_no_concat_support
def test_merge_scenes_with_transforms():
    """Test merge_scenes with transforms."""
    scene1 = create_test_scene(100, offset=(0, 0, 0), seed=1)
    scene2 = create_test_scene(100, offset=(0, 0, 0), seed=2)

    result = merge_scenes(
        [scene1, scene2],
        transforms=[
            Transform(),
            Transform().translate([2.0, 0.0, 0.0]),
        ],
    )

    assert len(result) == 200


@skip_no_concat_support
def test_merge_scenes_with_deduplication():
    """Test merge_scenes with deduplication."""
    # Create overlapping scenes
    scene1 = create_test_scene(100, offset=(0, 0, 0), seed=1)
    scene2 = create_test_scene(100, offset=(0, 0, 0), seed=1)  # Same position and seed

    result = merge_scenes([scene1, scene2], deduplicate_threshold=0.1)

    # Should have removed some duplicates
    assert len(result) < 200


def test_split_by_region():
    """Test splitting scene by spatial region."""
    # Create scene with Gaussians on left and right
    means = np.concatenate(
        [
            np.random.randn(50, 3).astype(np.float32) - [5, 0, 0],  # Left
            np.random.randn(50, 3).astype(np.float32) + [5, 0, 0],  # Right
        ]
    )

    n = len(means)
    scene = GSData(
        means=means,
        quats=np.tile([0, 0, 0, 1], (n, 1)).astype(np.float32),
        scales=np.ones((n, 3), dtype=np.float32) * 0.1,
        opacities=np.ones(n, dtype=np.float32),
        sh0=np.ones((n, 3), dtype=np.float32) * 0.5,
        shN=None,
    )

    # Split by X coordinate
    left, right = split_by_region(scene, lambda pos: pos[:, 0] < 0)

    assert len(left) + len(right) == len(scene)
    assert len(left) > 0
    assert len(right) > 0

    # Verify split
    assert np.all(left.means[:, 0] < 0)
    assert np.all(right.means[:, 0] >= 0)


def test_split_by_distance():
    """Test splitting scene by distance from origin."""
    # Create scene with near and far Gaussians
    means = np.concatenate(
        [
            np.random.randn(30, 3).astype(np.float32) * 0.5,  # Near origin
            (np.random.randn(70, 3).astype(np.float32) + [10, 10, 10]) * 2,  # Far
        ]
    )

    n = len(means)
    scene = GSData(
        means=means,
        quats=np.tile([0, 0, 0, 1], (n, 1)).astype(np.float32),
        scales=np.ones((n, 3), dtype=np.float32) * 0.1,
        opacities=np.ones(n, dtype=np.float32),
        sh0=np.ones((n, 3), dtype=np.float32) * 0.5,
        shN=None,
    )

    # Split by distance from origin
    near, far = split_by_region(scene, lambda pos: np.linalg.norm(pos, axis=1) < 5.0)

    assert len(near) + len(far) == len(scene)
    assert len(near) > 0
    assert len(far) > 0
