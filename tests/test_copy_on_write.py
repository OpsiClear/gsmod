"""
Tests for Copy-on-Write (COW) optimization in Color pipeline.

The COW optimization reduces memory usage when copying pipelines by sharing
the compiled LUT reference until modification is needed.
"""

import numpy as np
import pytest
from gsply import GSData

from gsmod import Color


@pytest.fixture
def sample_gsdata():
    """Create sample GSData for testing."""
    n = 1000
    data = GSData(
        means=np.random.randn(n, 3).astype(np.float32),
        scales=np.random.rand(n, 3).astype(np.float32),
        quats=np.random.randn(n, 4).astype(np.float32),
        opacities=np.random.rand(n).astype(np.float32),
        sh0=np.random.rand(n, 3).astype(np.float32),
        shN=None,
    )
    # Normalize quaternions
    data.quats[:] = data.quats / np.linalg.norm(data.quats, axis=1, keepdims=True)
    return data


class TestCopyOnWrite:
    """Test Copy-on-Write optimization."""

    def test_cow_shares_reference_on_copy(self):
        """Test that copying shares LUT reference instead of deep copying."""
        pipeline = Color().brightness(1.2).contrast(1.1)
        pipeline.compile()

        # Get reference to original LUT
        original_lut = pipeline._compiled_lut
        assert original_lut is not None

        # Copy pipeline
        pipeline2 = pipeline.copy()

        # Should share the same LUT reference
        assert pipeline2._compiled_lut is original_lut
        assert pipeline._lut_is_shared is True
        assert pipeline2._lut_is_shared is True

    def test_cow_trigger_on_compile(self):
        """Test that recompiling after copy triggers COW."""
        pipeline = Color().brightness(1.2).compile()
        original_lut = pipeline._compiled_lut

        # Copy and add new operation
        pipeline2 = pipeline.copy()
        assert pipeline2._compiled_lut is original_lut  # Initially shared

        # Add operation and compile (triggers COW)
        pipeline2.brightness(1.5).compile()

        # Should have different LUT now
        assert pipeline2._compiled_lut is not original_lut
        assert pipeline2._lut_is_shared is False
        assert pipeline._lut_is_shared is True  # Original still marked shared

    def test_cow_independence_after_trigger(self):
        """Test that modifications after COW don't affect original."""
        pipeline = Color().brightness(1.2).compile()

        # Copy and modify
        pipeline2 = pipeline.copy().contrast(2.0).compile()

        # Original should be unchanged
        assert len(pipeline._phase1_operations) == 1  # Only brightness
        assert len(pipeline2._phase1_operations) == 2  # Brightness + contrast

        # LUTs should be different
        assert not np.array_equal(pipeline._compiled_lut, pipeline2._compiled_lut)

    def test_cow_multiple_copies_share_same_lut(self):
        """Test that multiple copies can share the same LUT."""
        pipeline = Color().brightness(1.5).compile()
        original_lut = pipeline._compiled_lut

        # Create multiple copies
        copy1 = pipeline.copy()
        copy2 = pipeline.copy()
        copy3 = pipeline.copy()

        # All should share the same LUT reference
        assert copy1._compiled_lut is original_lut
        assert copy2._compiled_lut is original_lut
        assert copy3._compiled_lut is original_lut

        # All marked as shared
        assert pipeline._lut_is_shared is True
        assert copy1._lut_is_shared is True
        assert copy2._lut_is_shared is True
        assert copy3._lut_is_shared is True

    def test_cow_no_trigger_if_not_modified(self, sample_gsdata):
        """Test that using a copy without modification doesn't trigger COW."""
        pipeline = Color().brightness(1.2).compile()
        original_lut = pipeline._compiled_lut

        # Copy and use directly (no modifications)
        pipeline2 = pipeline.copy()
        result = pipeline2(sample_gsdata, inplace=False)

        # Should still share the same LUT
        assert pipeline2._compiled_lut is original_lut
        assert pipeline2._lut_is_shared is True

        # Results should be identical
        result_original = pipeline(sample_gsdata, inplace=False)
        assert np.allclose(result.sh0, result_original.sh0)

    def test_cow_reset_clears_shared_flag(self):
        """Test that reset() clears the shared flag."""
        pipeline = Color().brightness(1.2).compile()
        pipeline2 = pipeline.copy()

        # Both marked as shared
        assert pipeline._lut_is_shared is True
        assert pipeline2._lut_is_shared is True

        # Reset pipeline2
        pipeline2.reset()

        # Shared flag should be cleared
        assert pipeline2._lut_is_shared is False
        assert pipeline2._compiled_lut is None

    def test_cow_uncompiled_pipeline_copy(self):
        """Test copying an uncompiled pipeline."""
        pipeline = Color().brightness(1.2)  # Not compiled

        # Copy uncompiled pipeline
        pipeline2 = pipeline.copy()

        # No LUT to share
        assert pipeline._compiled_lut is None
        assert pipeline2._compiled_lut is None
        assert pipeline._lut_is_shared is False
        assert pipeline2._lut_is_shared is False

    def test_cow_cache_is_deep_copied(self):
        """Test that LUT cache is still deep copied (not shared)."""
        from gsmod import Param

        template = Color.template(brightness=Param("b", default=1.2, range=(0.5, 2.0)))

        # Build cache
        n = 100
        data = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.rand(n, 3).astype(np.float32),
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.rand(n).astype(np.float32),
            sh0=np.random.rand(n, 3).astype(np.float32),
            shN=None,
        )
        data.quats[:] = data.quats / np.linalg.norm(data.quats, axis=1, keepdims=True)
        template(data, params={"b": 1.5})

        # Copy template
        template2 = template.copy()

        # Cache should be deep copied (different references)
        assert template._lut_cache is not template2._lut_cache

        # But cache contents should be equal
        for key in template._lut_cache:
            assert key in template2._lut_cache
            # Arrays are copies, not shared
            assert template._lut_cache[key] is not template2._lut_cache[key]
            # But values are equal
            assert np.array_equal(template._lut_cache[key], template2._lut_cache[key])

    def test_cow_preserves_operations(self):
        """Test that COW preserves operation lists correctly."""
        pipeline = Color().brightness(1.2).contrast(1.1).saturation(1.3)
        pipeline.compile()

        # Copy
        pipeline2 = pipeline.copy()

        # Operations should be deep copied (independent)
        assert pipeline._phase1_operations == pipeline2._phase1_operations
        assert pipeline._phase1_operations is not pipeline2._phase1_operations

        # Modifying copy shouldn't affect original
        pipeline2.gamma(1.1)
        assert len(pipeline._phase1_operations) == 2  # brightness, contrast
        assert len(pipeline2._phase1_operations) == 3  # brightness, contrast, gamma


class TestCOWPerformance:
    """Test COW performance benefits."""

    def test_cow_memory_sharing(self):
        """Test that COW reduces memory usage by sharing LUT."""
        pipeline = Color().brightness(1.2).compile()

        # Create many copies
        copies = [pipeline.copy() for _ in range(100)]

        # All should share the same LUT (same memory address)
        original_id = id(pipeline._compiled_lut)
        for copy in copies:
            assert id(copy._compiled_lut) == original_id

        # Only 1 LUT in memory, not 101
        # (This is verified by the id() check above)

    def test_cow_with_parameterized_templates(self, sample_gsdata):
        """Test COW works correctly with parameterized templates."""
        from gsmod import Param

        template = Color.template(
            brightness=Param("b", default=1.2, range=(0.5, 2.0)),
            contrast=Param("c", default=1.1, range=(0.5, 2.0)),
        )

        # Use template with params
        result1 = template(sample_gsdata, inplace=False, params={"b": 1.5, "c": 1.2})

        # Copy template
        template2 = template.copy()

        # Use copy with same params (should use cache)
        result2 = template2(sample_gsdata, inplace=False, params={"b": 1.5, "c": 1.2})

        # Results should be identical
        assert np.allclose(result1.sh0, result2.sh0)

        # Cache should exist in both
        assert len(template._lut_cache) > 0
        assert len(template2._lut_cache) > 0
