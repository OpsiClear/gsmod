"""Transform operation configuration.

This module defines the standardized parameter specifications for all
transform operations, ensuring consistency between CPU and GPU pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass

from gsmod.config.operations import OperationSpec


@dataclass(frozen=True)
class TransformConfig:
    """Configuration for all transform pipeline operations.

    All parameters use standardized ranges that work identically
    for both CPU (Transform) and GPU (TransformGPU) pipelines.

    Note: Transform operations primarily use vector parameters (translation,
    rotation), so only scalar parameters are defined here.
    """

    scale_factor: OperationSpec = OperationSpec(
        name="scale_factor",
        min_value=0.001,
        max_value=1000.0,
        default=1.0,
        neutral=1.0,
        composition="multiplicative",
        description="Uniform scale multiplier: 1.0=no change",
    )

    rotation_angle: OperationSpec = OperationSpec(
        name="rotation_angle",
        min_value=-360.0,
        max_value=360.0,
        default=0.0,
        neutral=0.0,
        composition="additive",
        description="Rotation angle in degrees: 0=no rotation",
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
            "scale_factor": self.scale_factor,
            "rotation_angle": self.rotation_angle,
        }


# Singleton instance for use throughout the codebase
TRANSFORM_CONFIG = TransformConfig()
