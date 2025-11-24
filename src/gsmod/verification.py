"""Format verification utilities for CPU/GPU equivalence testing.

This module provides utilities for verifying that CPU and GPU pipelines
produce equivalent results with proper format tracking.

Example:
    >>> from gsmod.verification import FormatVerifier
    >>>
    >>> # After running CPU and GPU pipelines
    >>> FormatVerifier.assert_equivalent(cpu_result, gpu_result)
    >>>
    >>> # Check specific format
    >>> format = FormatVerifier.get_format(data)
    >>> if format != DataFormat.SH0_RGB:
    ...     data = data.to_rgb(inplace=True)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from gsply import GSData
from gsply.gsdata import DataFormat

if TYPE_CHECKING:
    from gsmod.torch import GSTensorPro

logger = logging.getLogger(__name__)


class FormatVerifier:
    """Utilities for verifying format equivalence between CPU and GPU."""

    @staticmethod
    def get_format(data: GSData | GSTensorPro) -> DataFormat:
        """Get the sh0 format of data.

        :param data: GSData or GSTensorPro object
        :return: DataFormat enum value (SH0_SH or SH0_RGB)

        Example:
            >>> format = FormatVerifier.get_format(data)
            >>> if format == DataFormat.SH0_RGB:
            ...     print("Data is in RGB format")
        """
        # Use gsply 0.2.8 format query properties
        if data.is_sh0_rgb:
            return DataFormat.SH0_RGB
        return DataFormat.SH0_SH

    @staticmethod
    def is_rgb(data: GSData | GSTensorPro) -> bool:
        """Check if data is in RGB format.

        :param data: GSData or GSTensorPro object
        :return: True if in RGB format, False otherwise
        """
        # Use gsply 0.2.8 format query property
        return data.is_sh0_rgb

    @staticmethod
    def is_sh(data: GSData | GSTensorPro) -> bool:
        """Check if data is in SH format.

        :param data: GSData or GSTensorPro object
        :return: True if in SH format, False otherwise
        """
        # Use gsply 0.2.8 format query property
        return data.is_sh0_sh

    @staticmethod
    def ensure_rgb(data: GSData | GSTensorPro, inplace: bool = True) -> GSData | GSTensorPro:
        """Ensure data is in RGB format, converting if necessary.

        :param data: GSData or GSTensorPro object
        :param inplace: If True, convert in-place
        :return: Data in RGB format

        Example:
            >>> data = FormatVerifier.ensure_rgb(data, inplace=True)
        """
        if not FormatVerifier.is_rgb(data):
            logger.debug("[FormatVerifier] Converting to RGB format")
            data = data.to_rgb(inplace=inplace)
        return data

    @staticmethod
    def ensure_sh(data: GSData | GSTensorPro, inplace: bool = True) -> GSData | GSTensorPro:
        """Ensure data is in SH format, converting if necessary.

        :param data: GSData or GSTensorPro object
        :param inplace: If True, convert in-place
        :return: Data in SH format

        Example:
            >>> data = FormatVerifier.ensure_sh(data, inplace=True)
        """
        if not FormatVerifier.is_sh(data):
            logger.debug("[FormatVerifier] Converting to SH format")
            data = data.to_sh(inplace=inplace)
        return data

    @staticmethod
    def assert_same_format(
        cpu_data: GSData,
        gpu_data: GSData,
        field: str = 'sh0'
    ) -> None:
        """Assert CPU and GPU data have same format.

        :param cpu_data: CPU result (GSData)
        :param gpu_data: GPU result (GSData)
        :param field: Field name to check (default: 'sh0')
        :raises AssertionError: If formats don't match

        Example:
            >>> FormatVerifier.assert_same_format(cpu_result, gpu_result)
        """
        cpu_format = FormatVerifier.get_format(cpu_data)
        gpu_format = FormatVerifier.get_format(gpu_data)

        if cpu_format != gpu_format:
            raise AssertionError(
                f"Format mismatch for {field}: CPU={cpu_format.name}, GPU={gpu_format.name}. "
                f"Both pipelines must produce same output format."
            )

    @staticmethod
    def assert_equivalent(
        cpu_result: GSData,
        gpu_result: GSData,
        rtol: float = 1e-5,
        atol: float = 1e-6,
        check_format: bool = True,
        check_all_fields: bool = True
    ) -> None:
        """Assert CPU and GPU results are equivalent with format verification.

        :param cpu_result: CPU pipeline result (GSData)
        :param gpu_result: GPU pipeline result (GSData)
        :param rtol: Relative tolerance for comparison
        :param atol: Absolute tolerance for comparison
        :param check_format: If True, verify formats match first
        :param check_all_fields: If True, check all fields (means, quats, scales, opacities)
        :raises AssertionError: If results are not equivalent

        Example:
            >>> FormatVerifier.assert_equivalent(
            ...     cpu_result, gpu_result,
            ...     rtol=1e-5, atol=1e-6
            ... )
        """
        # Step 1: Verify same format
        if check_format:
            FormatVerifier.assert_same_format(cpu_result, gpu_result)

        # Step 2: Verify same length
        if len(cpu_result) != len(gpu_result):
            raise AssertionError(
                f"Length mismatch: CPU={len(cpu_result)}, GPU={len(gpu_result)}"
            )

        # Step 3: Compare sh0 (color values)
        np.testing.assert_allclose(
            cpu_result.sh0,
            gpu_result.sh0,
            rtol=rtol,
            atol=atol,
            err_msg="Color values (sh0) differ between CPU and GPU"
        )

        if check_all_fields:
            # Step 4: Compare means (positions)
            np.testing.assert_allclose(
                cpu_result.means,
                gpu_result.means,
                rtol=rtol,
                atol=atol,
                err_msg="Positions (means) differ between CPU and GPU"
            )

            # Step 5: Compare quaternions
            np.testing.assert_allclose(
                cpu_result.quats,
                gpu_result.quats,
                rtol=rtol,
                atol=atol,
                err_msg="Quaternions differ between CPU and GPU"
            )

            # Step 6: Compare scales
            np.testing.assert_allclose(
                cpu_result.scales,
                gpu_result.scales,
                rtol=rtol,
                atol=atol,
                err_msg="Scales differ between CPU and GPU"
            )

            # Step 7: Compare opacities
            np.testing.assert_allclose(
                cpu_result.opacities,
                gpu_result.opacities,
                rtol=rtol,
                atol=atol,
                err_msg="Opacities differ between CPU and GPU"
            )

        logger.debug(
            "[FormatVerifier] Equivalence verified: %d Gaussians, format=%s",
            len(cpu_result),
            FormatVerifier.get_format(cpu_result).name
        )

    @staticmethod
    def format_summary(data: GSData | GSTensorPro) -> str:
        """Get a string summary of data format.

        :param data: GSData or GSTensorPro object
        :return: Format summary string

        Example:
            >>> print(FormatVerifier.format_summary(data))
            'sh0=RGB, n=1000'
        """
        format_val = FormatVerifier.get_format(data)
        format_name = "RGB" if format_val == DataFormat.SH0_RGB else "SH"
        return f"sh0={format_name}, n={len(data)}"
