"""
Protocol definitions for gsmod pipeline interfaces.

Defines the common interface that all pipeline stages must implement.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from gsply import GSData


@runtime_checkable
class PipelineStage(Protocol):
    """
    Protocol for pipeline stages (Color, Transform, Filter).

    All pipeline stages must implement this interface for compatibility
    with the unified Pipeline class.
    """

    def apply(self, data: GSData, inplace: bool = True) -> GSData:
        """
        Apply pipeline operations to GSData.

        :param data: GSData object to process
        :param inplace: If True, modify data in-place; if False, create copy
        :return: Processed GSData object
        """
        ...

    def reset(self) -> None:
        """Reset pipeline to initial state (no operations)."""
        ...

    def __call__(self, data: GSData, inplace: bool = True) -> GSData:
        """
        Apply pipeline operations (callable interface).

        :param data: GSData object to process
        :param inplace: If True, modify data in-place; if False, create copy
        :return: Processed GSData object
        """
        ...

    def __len__(self) -> int:
        """Return number of operations in pipeline."""
        ...
