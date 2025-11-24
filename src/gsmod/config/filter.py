"""Filter operation configuration.

This module defines the standardized parameter specifications for all
filter operations, ensuring consistency between CPU and GPU pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass

from gsmod.config.operations import OperationSpec


@dataclass(frozen=True)
class FilterConfig:
    """Configuration for all filter pipeline operations.

    All parameters use standardized ranges that work identically
    for both CPU (Filter) and GPU (FilterGPU) pipelines.
    """

    min_opacity: OperationSpec = OperationSpec(
        name="min_opacity",
        min_value=0.0,
        max_value=1.0,
        default=0.0,
        neutral=0.0,
        composition="additive",
        description="Minimum opacity threshold: keep Gaussians with opacity >= threshold",
    )

    max_scale: OperationSpec = OperationSpec(
        name="max_scale",
        min_value=0.0,
        max_value=100.0,
        default=100.0,
        neutral=100.0,
        composition="multiplicative",
        description="Maximum scale threshold: keep Gaussians with scale <= threshold",
    )

    sphere_radius: OperationSpec = OperationSpec(
        name="sphere_radius",
        min_value=0.0,
        max_value=1000.0,
        default=1.0,
        neutral=1000.0,
        composition="multiplicative",
        description="Sphere radius in absolute units (world coordinates)",
    )

    def get_spec(self, name: str) -> OperationSpec:
        """Get operation spec by name.

        :param name: Operation name
        :return: OperationSpec for the operation
        :raises AttributeError: If operation not found
        """
        return getattr(self, name)

    def get_all_specs(self) -> dict[str, OperationSpec]:
        """Get all operation specs as a dictionary.

        :return: Dictionary mapping operation names to specs
        """
        return {
            "min_opacity": self.min_opacity,
            "max_scale": self.max_scale,
            "sphere_radius": self.sphere_radius,
        }


# Singleton instance for use throughout the codebase
FILTER_CONFIG = FilterConfig()
