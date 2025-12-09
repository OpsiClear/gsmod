"""GPU-accelerated filtering pipeline for Gaussian Splatting."""

from __future__ import annotations

import numpy as np
import torch

from gsmod.torch.gstensor_pro import GSTensorPro


class FilterGPU:
    """GPU-accelerated filtering pipeline using PyTorch.

    Provides chainable filtering operations that run entirely on GPU with
    massive parallelism. All operations return self for fluent chaining.

    Performance:
        - 50-100x faster than CPU filtering
        - Parallel distance computations
        - Efficient boolean mask operations

    Example:
        >>> from gsmod.torch import FilterGPU
        >>> pipeline = (
        ...     FilterGPU()
        ...     .within_sphere(radius=1.0)
        ...     .min_opacity(0.1)
        ...     .max_scale(0.1)
        ... )
        >>> mask = pipeline.compute_mask(gstensor)
        >>> filtered = gstensor[mask]
    """

    def __init__(self):
        """Initialize filter pipeline."""
        self._operations = []

    def within_sphere(
        self, center: list[float] | np.ndarray | torch.Tensor = None, radius: float = 1.0
    ) -> FilterGPU:
        """Add spherical region filter.

        :param center: Sphere center [x, y, z] (default: origin)
        :param radius: Absolute radius in world units
        :returns: Self for chaining

        Example:
            >>> pipeline = FilterGPU().within_sphere(radius=5.0)  # 5.0 world units
        """
        self._operations.append(("within_sphere", (center, radius)))
        return self

    def within_box(
        self,
        min_bounds: list[float] | np.ndarray | torch.Tensor,
        max_bounds: list[float] | np.ndarray | torch.Tensor,
    ) -> FilterGPU:
        """Add axis-aligned box filter.

        :param min_bounds: Minimum bounds [x, y, z]
        :param max_bounds: Maximum bounds [x, y, z]
        :returns: Self for chaining

        Example:
            >>> pipeline = FilterGPU().within_box([-1, -1, -1], [1, 1, 1])
        """
        self._operations.append(("within_box", (min_bounds, max_bounds)))
        return self

    def min_opacity(self, threshold: float = 0.1) -> FilterGPU:
        """Add minimum opacity filter.

        :param threshold: Minimum opacity threshold
        :returns: Self for chaining

        Example:
            >>> pipeline = FilterGPU().min_opacity(0.1)
        """
        self._operations.append(("min_opacity", threshold))
        return self

    def max_opacity(self, threshold: float = 0.99) -> FilterGPU:
        """Add maximum opacity filter.

        :param threshold: Maximum opacity threshold
        :returns: Self for chaining

        Example:
            >>> pipeline = FilterGPU().max_opacity(0.99)
        """
        self._operations.append(("max_opacity", threshold))
        return self

    def min_scale(self, threshold: float = 0.001) -> FilterGPU:
        """Add minimum scale filter.

        :param threshold: Minimum scale threshold
        :returns: Self for chaining

        Example:
            >>> pipeline = FilterGPU().min_scale(0.001)
        """
        self._operations.append(("min_scale", threshold))
        return self

    def max_scale(self, threshold: float = 0.1) -> FilterGPU:
        """Add maximum scale filter.

        :param threshold: Maximum scale threshold
        :returns: Self for chaining

        Example:
            >>> pipeline = FilterGPU().max_scale(0.1)
        """
        self._operations.append(("max_scale", threshold))
        return self

    def outside_sphere(
        self, center: list[float] | np.ndarray | torch.Tensor = None, radius: float = 1.0
    ) -> FilterGPU:
        """Add filter for points outside sphere.

        :param center: Sphere center [x, y, z] (default: origin)
        :param radius: Sphere radius
        :returns: Self for chaining

        Example:
            >>> pipeline = FilterGPU().outside_sphere(radius=0.5)
        """
        self._operations.append(("outside_sphere", (center, radius)))
        return self

    def outside_box(
        self,
        min_bounds: list[float] | np.ndarray | torch.Tensor,
        max_bounds: list[float] | np.ndarray | torch.Tensor,
    ) -> FilterGPU:
        """Add filter for points outside box.

        :param min_bounds: Minimum bounds [x, y, z]
        :param max_bounds: Maximum bounds [x, y, z]
        :returns: Self for chaining

        Example:
            >>> pipeline = FilterGPU().outside_box([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
        """
        self._operations.append(("outside_box", (min_bounds, max_bounds)))
        return self

    def within_rotated_box(
        self,
        center: list[float] | np.ndarray | torch.Tensor,
        size: list[float] | np.ndarray | torch.Tensor,
        rotation: list[float] | np.ndarray | torch.Tensor | None = None,
    ) -> FilterGPU:
        """Add rotated box (OBB) filter.

        :param center: Box center [x, y, z]
        :param size: Box full size [width, height, depth]
        :param rotation: Rotation as axis-angle [rx, ry, rz] in radians, or None
        :returns: Self for chaining

        Example:
            >>> import numpy as np
            >>> # Box rotated 45 degrees around Y axis
            >>> pipeline = FilterGPU().within_rotated_box(
            ...     center=[0, 0, 0],
            ...     size=[2, 2, 2],
            ...     rotation=[0, np.pi/4, 0]
            ... )
        """
        self._operations.append(("within_rotated_box", (center, size, rotation)))
        return self

    def outside_rotated_box(
        self,
        center: list[float] | np.ndarray | torch.Tensor,
        size: list[float] | np.ndarray | torch.Tensor,
        rotation: list[float] | np.ndarray | torch.Tensor | None = None,
    ) -> FilterGPU:
        """Add filter for points outside rotated box (OBB).

        :param center: Box center [x, y, z]
        :param size: Box full size [width, height, depth]
        :param rotation: Rotation as axis-angle [rx, ry, rz] in radians, or None
        :returns: Self for chaining

        Example:
            >>> pipeline = FilterGPU().outside_rotated_box(
            ...     center=[0, 0, 0],
            ...     size=[1, 1, 1],
            ...     rotation=[0, 0.5, 0]
            ... )
        """
        self._operations.append(("outside_rotated_box", (center, size, rotation)))
        return self

    def within_ellipsoid(
        self,
        center: list[float] | np.ndarray | torch.Tensor = None,
        radii: list[float] | np.ndarray | torch.Tensor = None,
        rotation: list[float] | np.ndarray | torch.Tensor | None = None,
    ) -> FilterGPU:
        """Add ellipsoid filter.

        :param center: Ellipsoid center [x, y, z] (default: origin)
        :param radii: Ellipsoid radii [rx, ry, rz] (default: [1, 1, 1])
        :param rotation: Rotation as axis-angle [rx, ry, rz] in radians, or None
        :returns: Self for chaining

        Example:
            >>> # Ellipsoid stretched along X axis, rotated 30 deg around Z
            >>> pipeline = FilterGPU().within_ellipsoid(
            ...     center=[0, 0, 0],
            ...     radii=[3.0, 1.0, 1.0],
            ...     rotation=[0, 0, 0.52]
            ... )
        """
        self._operations.append(("within_ellipsoid", (center, radii, rotation)))
        return self

    def outside_ellipsoid(
        self,
        center: list[float] | np.ndarray | torch.Tensor = None,
        radii: list[float] | np.ndarray | torch.Tensor = None,
        rotation: list[float] | np.ndarray | torch.Tensor | None = None,
    ) -> FilterGPU:
        """Add filter for points outside ellipsoid.

        :param center: Ellipsoid center [x, y, z] (default: origin)
        :param radii: Ellipsoid radii [rx, ry, rz] (default: [1, 1, 1])
        :param rotation: Rotation as axis-angle [rx, ry, rz] in radians, or None
        :returns: Self for chaining

        Example:
            >>> pipeline = FilterGPU().outside_ellipsoid(
            ...     center=[0, 0, 0],
            ...     radii=[2.0, 1.0, 1.0]
            ... )
        """
        self._operations.append(("outside_ellipsoid", (center, radii, rotation)))
        return self

    def within_frustum(
        self,
        position: list[float] | np.ndarray | torch.Tensor = None,
        rotation: list[float] | np.ndarray | torch.Tensor | None = None,
        fov: float = 1.047,
        aspect: float = 1.0,
        near: float = 0.1,
        far: float = 100.0,
    ) -> FilterGPU:
        """Add camera frustum filter.

        Camera convention: -Z is forward, +X is right, +Y is up.

        :param position: Camera position [x, y, z] (default: origin)
        :param rotation: Camera rotation as axis-angle [rx, ry, rz] in radians, or None
        :param fov: Vertical field of view in radians (default: 60 degrees)
        :param aspect: Aspect ratio width/height (default: 1.0)
        :param near: Near clipping plane distance (default: 0.1)
        :param far: Far clipping plane distance (default: 100.0)
        :returns: Self for chaining

        Example:
            >>> import numpy as np
            >>> # Frustum looking down -Z with 90 degree FOV
            >>> pipeline = FilterGPU().within_frustum(
            ...     position=[0, 0, 10],
            ...     rotation=None,
            ...     fov=np.pi/2,
            ...     aspect=16/9,
            ...     near=0.1,
            ...     far=100.0
            ... )
        """
        self._operations.append(("within_frustum", (position, rotation, fov, aspect, near, far)))
        return self

    def outside_frustum(
        self,
        position: list[float] | np.ndarray | torch.Tensor = None,
        rotation: list[float] | np.ndarray | torch.Tensor | None = None,
        fov: float = 1.047,
        aspect: float = 1.0,
        near: float = 0.1,
        far: float = 100.0,
    ) -> FilterGPU:
        """Add filter for points outside camera frustum.

        :param position: Camera position [x, y, z] (default: origin)
        :param rotation: Camera rotation as axis-angle [rx, ry, rz] in radians, or None
        :param fov: Vertical field of view in radians (default: 60 degrees)
        :param aspect: Aspect ratio width/height (default: 1.0)
        :param near: Near clipping plane distance (default: 0.1)
        :param far: Far clipping plane distance (default: 100.0)
        :returns: Self for chaining

        Example:
            >>> pipeline = FilterGPU().outside_frustum(
            ...     position=[0, 0, 10],
            ...     fov=1.047,
            ...     near=0.1,
            ...     far=50.0
            ... )
        """
        self._operations.append(("outside_frustum", (position, rotation, fov, aspect, near, far)))
        return self

    def near_plane(self, distance: float = 0.1, axis: str = "z") -> FilterGPU:
        """Add near plane filter (keep points beyond distance).

        :param distance: Distance from origin along axis
        :param axis: Axis ("x", "y", or "z")
        :returns: Self for chaining

        Example:
            >>> pipeline = FilterGPU().near_plane(0.1, axis="z")
        """
        self._operations.append(("near_plane", (distance, axis)))
        return self

    def far_plane(self, distance: float = 10.0, axis: str = "z") -> FilterGPU:
        """Add far plane filter (keep points within distance).

        :param distance: Distance from origin along axis
        :param axis: Axis ("x", "y", or "z")
        :returns: Self for chaining

        Example:
            >>> pipeline = FilterGPU().far_plane(10.0, axis="z")
        """
        self._operations.append(("far_plane", (distance, axis)))
        return self

    def compute_mask(self, data: GSTensorPro, mode: str = "and") -> torch.Tensor:
        """Compute filter mask for GSTensorPro.

        :param data: GSTensorPro object to filter
        :param mode: Combination mode ("and" = all conditions, "or" = any condition)
        :returns: Boolean mask tensor

        Example:
            >>> pipeline = FilterGPU().within_sphere(1.0).min_opacity(0.1)
            >>> mask = pipeline.compute_mask(gstensor)
            >>> filtered = gstensor[mask]
        """
        if not isinstance(data, GSTensorPro):
            raise TypeError(f"Expected GSTensorPro, got {type(data)}")

        if not self._operations:
            # No filters - return all True
            return torch.ones(len(data), dtype=torch.bool, device=data.device)

        masks = []

        # Apply each filter
        for op_name, op_value in self._operations:
            if op_name == "within_sphere":
                center, radius = op_value
                mask = data.filter_within_sphere(center, radius)
            elif op_name == "within_box":
                min_bounds, max_bounds = op_value
                mask = data.filter_within_box(min_bounds, max_bounds)
            elif op_name == "min_opacity":
                mask = data.filter_min_opacity(op_value)
            elif op_name == "max_opacity":
                mask = self._filter_max_opacity(data, op_value)
            elif op_name == "min_scale":
                mask = self._filter_min_scale(data, op_value)
            elif op_name == "max_scale":
                mask = data.filter_max_scale(op_value)
            elif op_name == "outside_sphere":
                center, radius = op_value
                # Invert within_sphere
                mask = ~data.filter_within_sphere(center, radius)
            elif op_name == "outside_box":
                min_bounds, max_bounds = op_value
                # Invert within_box
                mask = ~data.filter_within_box(min_bounds, max_bounds)
            elif op_name == "near_plane":
                distance, axis = op_value
                mask = self._filter_plane(data, distance, axis, "near")
            elif op_name == "far_plane":
                distance, axis = op_value
                mask = self._filter_plane(data, distance, axis, "far")
            elif op_name == "within_rotated_box":
                center, size, rotation = op_value
                mask = self._filter_rotated_box(data, center, size, rotation)
            elif op_name == "outside_rotated_box":
                center, size, rotation = op_value
                mask = ~self._filter_rotated_box(data, center, size, rotation)
            elif op_name == "within_ellipsoid":
                center, radii, rotation = op_value
                mask = self._filter_ellipsoid(data, center, radii, rotation)
            elif op_name == "outside_ellipsoid":
                center, radii, rotation = op_value
                mask = ~self._filter_ellipsoid(data, center, radii, rotation)
            elif op_name == "within_frustum":
                position, rotation, fov, aspect, near, far = op_value
                mask = self._filter_frustum(data, position, rotation, fov, aspect, near, far)
            elif op_name == "outside_frustum":
                position, rotation, fov, aspect, near, far = op_value
                mask = ~self._filter_frustum(data, position, rotation, fov, aspect, near, far)
            else:
                raise ValueError(f"Unknown filter operation: {op_name}")

            masks.append(mask)

        # Combine masks
        if mode == "and":
            # All conditions must be True
            result_mask = masks[0]
            for mask in masks[1:]:
                result_mask = result_mask & mask
        elif mode == "or":
            # Any condition must be True
            result_mask = masks[0]
            for mask in masks[1:]:
                result_mask = result_mask | mask
        else:
            raise ValueError(f"Mode must be 'and' or 'or', got '{mode}'")

        return result_mask

    def __call__(self, data: GSTensorPro, mode: str = "and", inplace: bool = True) -> GSTensorPro:
        """Apply filter pipeline to GSTensorPro.

        :param data: GSTensorPro object to filter
        :param mode: Combination mode ("and" = all conditions, "or" = any condition)
        :param inplace: If True, modify data in-place; if False, return filtered copy
        :returns: Filtered GSTensorPro

        Example:
            >>> pipeline = FilterGPU().within_sphere(1.0).min_opacity(0.1)
            >>> filtered = pipeline(gstensor, mode="and", inplace=False)
        """
        mask = self.compute_mask(data, mode)

        if inplace:
            # Filter in-place using mask
            data.means = data.means[mask]
            data.scales = data.scales[mask]
            data.quats = data.quats[mask]
            data.opacities = data.opacities[mask]
            data.sh0 = data.sh0[mask]
            if data.shN is not None:
                data.shN = data.shN[mask]
            if data.masks is not None:
                data.masks = data.masks[mask]
            if data._base is not None:
                data._base = data._base[mask]
            return data

        # Return filtered copy
        return data[mask]

    def _filter_max_opacity(self, data: GSTensorPro, threshold: float) -> torch.Tensor:
        """Filter Gaussians with maximum opacity.

        :param data: GSTensorPro
        :param threshold: Maximum opacity threshold
        :returns: Boolean mask
        """
        # Handle different opacity formats using gsply's is_opacities_ply property
        if data.is_opacities_ply:
            # Logit opacities - convert threshold to logit
            threshold_logit = torch.logit(torch.tensor(threshold, device=data.device))
            return data.opacities <= threshold_logit
        else:
            # Linear opacities
            return data.opacities <= threshold

    def _filter_min_scale(self, data: GSTensorPro, threshold: float) -> torch.Tensor:
        """Filter Gaussians with minimum scale.

        :param data: GSTensorPro
        :param threshold: Minimum scale threshold
        :returns: Boolean mask
        """
        # Handle different scale formats using gsply's is_scales_ply property
        if data.is_scales_ply:
            # Log scales - convert threshold to log
            threshold_log = torch.log(torch.tensor(threshold, device=data.device))
            min_scales = torch.min(data.scales, dim=1)[0]
            return min_scales >= threshold_log
        else:
            # Linear scales
            min_scales = torch.min(data.scales, dim=1)[0]
            return min_scales >= threshold

    def _filter_plane(
        self, data: GSTensorPro, distance: float, axis: str, plane_type: str
    ) -> torch.Tensor:
        """Filter by plane (near or far).

        :param data: GSTensorPro
        :param distance: Distance from origin
        :param axis: Axis ("x", "y", or "z")
        :param plane_type: "near" or "far"
        :returns: Boolean mask
        """
        axis_map = {"x": 0, "y": 1, "z": 2}
        if axis not in axis_map:
            raise ValueError(f"Axis must be 'x', 'y', or 'z', got '{axis}'")

        axis_idx = axis_map[axis]
        positions = data.means[:, axis_idx]

        if plane_type == "near":
            # Keep points beyond distance
            return positions >= distance
        else:  # far
            # Keep points within distance
            return positions <= distance

    def _axis_angle_to_rotation_matrix(
        self,
        axis_angle: list[float] | np.ndarray | torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert axis-angle rotation to 3x3 rotation matrix.

        :param axis_angle: Rotation vector [3] where magnitude is angle in radians
        :param device: Target device
        :param dtype: Target dtype
        :returns: Rotation matrix [3, 3]
        """
        if not isinstance(axis_angle, torch.Tensor):
            axis_angle = torch.tensor(axis_angle, dtype=dtype, device=device)
        else:
            axis_angle = axis_angle.to(dtype=dtype, device=device)

        angle = torch.norm(axis_angle)

        if angle < 1e-8:
            return torch.eye(3, dtype=dtype, device=device)

        # Normalize axis
        axis = axis_angle / angle

        # Rodrigues' rotation formula
        K = torch.tensor(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
            dtype=dtype,
            device=device,
        )

        # R = I + sin(angle) * K + (1 - cos(angle)) * K^2
        R = (
            torch.eye(3, dtype=dtype, device=device)
            + torch.sin(angle) * K
            + (1 - torch.cos(angle)) * (K @ K)
        )

        return R

    def _filter_rotated_box(
        self,
        data: GSTensorPro,
        center: list[float] | np.ndarray | torch.Tensor,
        size: list[float] | np.ndarray | torch.Tensor,
        rotation: list[float] | np.ndarray | torch.Tensor | None,
    ) -> torch.Tensor:
        """Filter Gaussians within rotated box (OBB).

        :param data: GSTensorPro
        :param center: Box center [x, y, z]
        :param size: Box full size [width, height, depth]
        :param rotation: Rotation as axis-angle [rx, ry, rz] in radians, or None
        :returns: Boolean mask
        """
        device = data.device
        dtype = data.dtype

        # Convert center and compute half extents
        if not isinstance(center, torch.Tensor):
            center = torch.tensor(center, dtype=dtype, device=device)
        else:
            center = center.to(dtype=dtype, device=device)

        if not isinstance(size, torch.Tensor):
            size = torch.tensor(size, dtype=dtype, device=device)
        else:
            size = size.to(dtype=dtype, device=device)

        half_extents = size * 0.5

        # Get rotation matrix
        if rotation is not None:
            rot_matrix = self._axis_angle_to_rotation_matrix(rotation, device, dtype)
            rot_matrix = rot_matrix.T  # Transpose for world-to-local
        else:
            rot_matrix = torch.eye(3, dtype=dtype, device=device)

        # Transform to local coordinates
        delta = data.means - center
        local = delta @ rot_matrix.T

        # Check if inside box (abs local coords <= half extents)
        mask = torch.all(torch.abs(local) <= half_extents, dim=1)

        return mask

    def _filter_ellipsoid(
        self,
        data: GSTensorPro,
        center: list[float] | np.ndarray | torch.Tensor | None,
        radii: list[float] | np.ndarray | torch.Tensor | None,
        rotation: list[float] | np.ndarray | torch.Tensor | None,
    ) -> torch.Tensor:
        """Filter Gaussians within ellipsoid.

        :param data: GSTensorPro
        :param center: Ellipsoid center [x, y, z] (default: origin)
        :param radii: Ellipsoid radii [rx, ry, rz] (default: [1, 1, 1])
        :param rotation: Rotation as axis-angle [rx, ry, rz] in radians, or None
        :returns: Boolean mask
        """
        device = data.device
        dtype = data.dtype

        # Default center to origin
        if center is None:
            center = torch.zeros(3, dtype=dtype, device=device)
        elif not isinstance(center, torch.Tensor):
            center = torch.tensor(center, dtype=dtype, device=device)
        else:
            center = center.to(dtype=dtype, device=device)

        # Default radii to unit sphere
        if radii is None:
            radii = torch.ones(3, dtype=dtype, device=device)
        elif not isinstance(radii, torch.Tensor):
            radii = torch.tensor(radii, dtype=dtype, device=device)
        else:
            radii = radii.to(dtype=dtype, device=device)

        # Get rotation matrix
        if rotation is not None:
            rot_matrix = self._axis_angle_to_rotation_matrix(rotation, device, dtype)
            rot_matrix = rot_matrix.T  # Transpose for world-to-local
        else:
            rot_matrix = torch.eye(3, dtype=dtype, device=device)

        # Transform to local coordinates
        delta = data.means - center
        local = delta @ rot_matrix.T

        # Normalize by radii and compute ellipsoid distance
        # Guard against division by zero with minimum radius
        normalized = local / torch.clamp(radii, min=1e-7)
        dist_sq = torch.sum(normalized**2, dim=1)

        # Point is inside if normalized distance <= 1
        mask = dist_sq <= 1.0

        return mask

    def _filter_frustum(
        self,
        data: GSTensorPro,
        position: list[float] | np.ndarray | torch.Tensor | None,
        rotation: list[float] | np.ndarray | torch.Tensor | None,
        fov: float,
        aspect: float,
        near: float,
        far: float,
    ) -> torch.Tensor:
        """Filter Gaussians within camera frustum.

        Camera convention: -Z is forward, +X is right, +Y is up.

        :param data: GSTensorPro
        :param position: Camera position [x, y, z] (default: origin)
        :param rotation: Camera rotation as axis-angle [rx, ry, rz] in radians, or None
        :param fov: Vertical field of view in radians
        :param aspect: Aspect ratio width/height
        :param near: Near clipping plane distance
        :param far: Far clipping plane distance
        :returns: Boolean mask
        """
        device = data.device
        dtype = data.dtype

        # Default position to origin
        if position is None:
            camera_pos = torch.zeros(3, dtype=dtype, device=device)
        elif not isinstance(position, torch.Tensor):
            camera_pos = torch.tensor(position, dtype=dtype, device=device)
        else:
            camera_pos = position.to(dtype=dtype, device=device)

        # Get rotation matrix
        if rotation is not None:
            rot_matrix = self._axis_angle_to_rotation_matrix(rotation, device, dtype)
            rot_matrix = rot_matrix.T  # Transpose for world-to-camera
        else:
            rot_matrix = torch.eye(3, dtype=dtype, device=device)

        # Transform to camera coordinates
        delta = data.means - camera_pos
        local = delta @ rot_matrix.T

        # Calculate FOV tangents
        tan_half_fov_y = torch.tan(torch.tensor(fov / 2, device=device))
        tan_half_fov_x = tan_half_fov_y * aspect

        # Check frustum bounds (camera looks down -Z)
        lx = local[:, 0]
        ly = local[:, 1]
        lz = local[:, 2]

        # Depth check: must be in front of camera and within near/far
        in_depth = (lz < -near) & (lz > -far)

        # FOV check using absolute Z (since -Z is forward)
        abs_z = -lz
        in_fov_x = torch.abs(lx) <= abs_z * tan_half_fov_x
        in_fov_y = torch.abs(ly) <= abs_z * tan_half_fov_y

        # Combine all conditions
        mask = in_depth & in_fov_x & in_fov_y

        return mask

    def reset(self) -> FilterGPU:
        """Reset pipeline, removing all operations.

        :returns: Self for chaining

        Example:
            >>> pipeline.reset().within_sphere(1.0)  # Clear and start fresh
        """
        self._operations = []
        return self

    def clone(self) -> FilterGPU:
        """Create a copy of this pipeline.

        :returns: New FilterGPU with same operations

        Example:
            >>> pipeline2 = pipeline1.clone()
            >>> pipeline2.min_opacity(0.2)  # Doesn't affect pipeline1
        """
        new_pipeline = FilterGPU()
        new_pipeline._operations = self._operations.copy()
        return new_pipeline

    def invert(self) -> FilterGPU:
        """Create inverted filter pipeline.

        Useful for selecting complement of filtered region.

        :returns: New FilterGPU with inverted logic

        Example:
            >>> # Select points outside sphere with low opacity
            >>> pipeline = FilterGPU().within_sphere(1.0).min_opacity(0.1)
            >>> inverted = pipeline.invert()
            >>> # Inverted selects: outside sphere OR opacity < 0.1
        """
        new_pipeline = FilterGPU()
        # Invert each operation
        for op_name, op_value in self._operations:
            if op_name == "within_sphere":
                new_pipeline._operations.append(("outside_sphere", op_value))
            elif op_name == "outside_sphere":
                new_pipeline._operations.append(("within_sphere", op_value))
            elif op_name == "within_box":
                new_pipeline._operations.append(("outside_box", op_value))
            elif op_name == "outside_box":
                new_pipeline._operations.append(("within_box", op_value))
            elif op_name == "within_rotated_box":
                new_pipeline._operations.append(("outside_rotated_box", op_value))
            elif op_name == "outside_rotated_box":
                new_pipeline._operations.append(("within_rotated_box", op_value))
            elif op_name == "within_ellipsoid":
                new_pipeline._operations.append(("outside_ellipsoid", op_value))
            elif op_name == "outside_ellipsoid":
                new_pipeline._operations.append(("within_ellipsoid", op_value))
            elif op_name == "within_frustum":
                new_pipeline._operations.append(("outside_frustum", op_value))
            elif op_name == "outside_frustum":
                new_pipeline._operations.append(("within_frustum", op_value))
            elif op_name == "min_opacity":
                # min_opacity >= threshold becomes < threshold
                new_pipeline._operations.append(("max_opacity", op_value - 1e-6))
            elif op_name == "max_opacity":
                # max_opacity <= threshold becomes > threshold
                new_pipeline._operations.append(("min_opacity", op_value + 1e-6))
            elif op_name == "min_scale":
                new_pipeline._operations.append(("max_scale", op_value - 1e-6))
            elif op_name == "max_scale":
                new_pipeline._operations.append(("min_scale", op_value + 1e-6))
            elif op_name == "near_plane":
                # Positions >= distance becomes < distance
                distance, axis = op_value
                new_pipeline._operations.append(("far_plane", (distance - 1e-6, axis)))
            elif op_name == "far_plane":
                # Positions <= distance becomes > distance
                distance, axis = op_value
                new_pipeline._operations.append(("near_plane", (distance + 1e-6, axis)))

        return new_pipeline
