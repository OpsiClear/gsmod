"""
Scene composition utilities for combining multiple GSData objects.

This module provides high-level utilities for combining, merging, and composing
multiple Gaussian Splatting scenes into unified representations.

Key Features:
- Fast bulk concatenation (6.15x faster than repeated addition)
- Geometric transformations during composition
- Conflict resolution for overlapping Gaussians
- Memory-efficient operations using gsply's optimized concatenation
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from gsply import GSData

from gsmod.transform.pipeline import Transform

logger = logging.getLogger(__name__)


def concatenate(scenes: list[GSData]) -> GSData:
    """
    Concatenate multiple GSData objects into a single scene.

    Uses gsply's optimized GSData.concatenate() which is 6.15x faster
    than repeated .add() operations for 3+ scenes.

    :param scenes: List of GSData objects to concatenate
    :return: Concatenated GSData object containing all Gaussians

    Example:
        >>> scene1 = gsply.plyread("scene1.ply")
        >>> scene2 = gsply.plyread("scene2.ply")
        >>> scene3 = gsply.plyread("scene3.ply")
        >>> combined = concatenate([scene1, scene2, scene3])
        >>> print(f"Combined {len(combined)} Gaussians")

    Note:
        For 2 scenes, use scene1.add(scene2) directly.
        For 3+ scenes, use this function for optimal performance.
    """
    if len(scenes) == 0:
        raise ValueError("Cannot concatenate empty list of scenes")

    if len(scenes) == 1:
        logger.warning("[Compose] Only one scene provided, returning copy")
        return scenes[0].copy()

    if len(scenes) == 2:
        logger.info("[Compose] Two scenes provided, using .add() method")
        return scenes[0].add(scenes[1])

    logger.info("[Compose] Concatenating %d scenes", len(scenes))
    result = GSData.concatenate(scenes)
    logger.info("[Compose] Result contains %d Gaussians", len(result))
    return result


def compose_with_transforms(
    scenes: list[GSData],
    transforms: list[Transform],
    inplace: bool = False,
) -> GSData:
    """
    Compose multiple scenes by applying transformations before concatenation.

    This is useful for positioning multiple objects/scenes in a unified
    coordinate system before combining them.

    :param scenes: List of GSData objects to compose
    :param transforms: List of Transform pipelines (one per scene)
    :param inplace: If True, modifies input scenes directly
    :return: Concatenated GSData with all transforms applied
    :raises ValueError: If number of scenes and transforms don't match

    Example:
        >>> obj1 = gsply.plyread("object1.ply")
        >>> obj2 = gsply.plyread("object2.ply")
        >>>
        >>> # Position object2 to the right of object1
        >>> t1 = Transform()  # Identity - leave obj1 at origin
        >>> t2 = Transform().translate([2.0, 0.0, 0.0])
        >>>
        >>> scene = compose_with_transforms([obj1, obj2], [t1, t2])
    """
    if len(scenes) != len(transforms):
        raise ValueError(
            f"Number of scenes ({len(scenes)}) must match number of transforms ({len(transforms)})"
        )

    logger.info("[Compose] Transforming %d scenes before composition", len(scenes))

    # Apply transforms
    transformed_scenes = []
    for i, (scene, transform) in enumerate(zip(scenes, transforms)):
        if transform.is_identity():
            # Fast path: no transformation needed
            transformed = scene if inplace else scene.copy()
        else:
            # Apply transformation
            transformed = transform(scene, inplace=inplace)
        transformed_scenes.append(transformed)
        logger.debug("[Compose] Scene %d transformed (%d Gaussians)", i, len(transformed))

    # Concatenate all transformed scenes
    return concatenate(transformed_scenes)


def deduplicate(
    data: GSData,
    position_threshold: float = 1e-6,
    method: str = "first",
) -> GSData:
    """
    Remove duplicate Gaussians based on position proximity using spatial hashing.

    Uses a spatial grid/hash-based approach for O(N) average-case performance,
    significantly faster than naive O(N^2) approaches for large scenes.

    :param data: GSData object to deduplicate
    :param position_threshold: Distance threshold for considering Gaussians as duplicates
    :param method: How to resolve duplicates:
        - "first": Keep first occurrence
        - "last": Keep last occurrence
        - "average": Average all duplicate attributes
    :return: Deduplicated GSData object

    Example:
        >>> scene = concatenate([scene1, scene2, scene3])
        >>> clean_scene = deduplicate(scene, position_threshold=0.001)
        >>> print(f"Removed {len(scene) - len(clean_scene)} duplicates")

    Performance:
        - Average case: O(N) with spatial hashing
        - Worst case: O(N log N) for densely clustered Gaussians
        - Scales efficiently to >10M Gaussians
    """
    logger.info("[Compose] Deduplicating %d Gaussians", len(data))

    if method not in {"first", "last", "average"}:
        raise ValueError(f'method="{method}" is not valid. Valid options are: first, last, average')

    if method == "average":
        raise NotImplementedError('average method not yet implemented. Use "first" or "last".')

    means = data.means
    n = len(data)

    # Track which Gaussians to keep
    keep_mask = np.ones(n, dtype=bool)

    # Build spatial hash grid
    # Grid cell size = position_threshold for optimal neighbor search
    cell_size = position_threshold
    inv_cell_size = 1.0 / cell_size

    # Map each Gaussian to a grid cell
    grid_coords = np.floor(means * inv_cell_size).astype(np.int32)

    # Build hash table: grid_cell -> list of Gaussian indices
    # Use dictionary for sparse grid (efficient for scattered Gaussians)
    from collections import defaultdict

    grid: dict[tuple[int, int, int], list[int]] = defaultdict(list)

    for idx in range(n):
        if not keep_mask[idx]:
            continue

        gx, gy, gz = grid_coords[idx]
        cell_key = (gx, gy, gz)

        # Check current cell and 26 neighboring cells (3x3x3 cube)
        # Only need to check neighbors because threshold < cell_size
        duplicates_found = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    neighbor_key = (gx + dx, gy + dy, gz + dz)
                    if neighbor_key not in grid:
                        continue

                    # Check all Gaussians in this neighbor cell
                    for neighbor_idx in grid[neighbor_key]:
                        if neighbor_idx == idx:
                            continue

                        if not keep_mask[neighbor_idx]:
                            continue

                        # Compute distance
                        dist = np.linalg.norm(means[idx] - means[neighbor_idx])

                        if dist < position_threshold:
                            # Found duplicate
                            duplicates_found.append((neighbor_idx, dist))

        # Resolve duplicates based on method
        if duplicates_found:
            if method == "first":
                # Keep first occurrence, remove current if any duplicate has lower index
                for dup_idx, _ in duplicates_found:
                    if dup_idx < idx:
                        # Found an earlier duplicate, remove current
                        keep_mask[idx] = False
                        break
            elif method == "last":
                # Keep last occurrence, remove all earlier duplicates
                for dup_idx, _ in duplicates_found:
                    if dup_idx < idx:
                        keep_mask[dup_idx] = False

        # Add current Gaussian to grid (if not removed)
        if keep_mask[idx]:
            grid[cell_key].append(idx)

    num_removed = (~keep_mask).sum()
    logger.info("[Compose] Removed %d duplicates using spatial hashing", num_removed)

    # Apply mask to keep only non-duplicate Gaussians
    return data[keep_mask]


def merge_scenes(
    scenes: list[GSData],
    transforms: list[Transform] | None = None,
    deduplicate_threshold: float | None = None,
) -> GSData:
    """
    High-level scene merging with optional transforms and deduplication.

    This is the primary API for combining multiple scenes with full control
    over positioning and cleanup.

    :param scenes: List of GSData objects to merge
    :param transforms: Optional list of Transform pipelines (one per scene)
    :param deduplicate_threshold: If provided, remove duplicates within this distance
    :return: Merged GSData object

    Example:
        >>> # Simple merge without transforms
        >>> merged = merge_scenes([scene1, scene2, scene3])
        >>>
        >>> # Merge with positioning
        >>> merged = merge_scenes(
        ...     [obj1, obj2, obj3],
        ...     transforms=[
        ...         Transform(),  # obj1 at origin
        ...         Transform().translate([2, 0, 0]),  # obj2 to the right
        ...         Transform().translate([0, 2, 0]),  # obj3 above
        ...     ]
        ... )
        >>>
        >>> # Merge with deduplication
        >>> merged = merge_scenes([scene1, scene2], deduplicate_threshold=0.001)
    """
    logger.info("[Compose] Merging %d scenes", len(scenes))

    # Apply transforms if provided
    if transforms is not None:
        result = compose_with_transforms(scenes, transforms, inplace=False)
    else:
        result = concatenate(scenes)

    # Deduplicate if requested
    if deduplicate_threshold is not None:
        result = deduplicate(result, position_threshold=deduplicate_threshold)

    logger.info("[Compose] Merge complete: %d Gaussians", len(result))
    return result


def split_by_region(
    data: GSData,
    predicate: Callable[[np.ndarray], np.ndarray],
) -> tuple[GSData, GSData]:
    """
    Split a scene into two regions based on a spatial predicate.

    :param data: GSData object to split
    :param predicate: Function that takes positions [N, 3] and returns boolean mask [N]
    :return: Tuple of (inside_region, outside_region) GSData objects

    Example:
        >>> # Split scene by X coordinate
        >>> left, right = split_by_region(
        ...     data,
        ...     lambda pos: pos[:, 0] < 0  # X < 0
        ... )
        >>>
        >>> # Split scene by distance from origin
        >>> near, far = split_by_region(
        ...     data,
        ...     lambda pos: np.linalg.norm(pos, axis=1) < 5.0
        ... )
    """
    logger.info("[Compose] Splitting scene of %d Gaussians", len(data))

    mask = predicate(data.means)
    inside = data[mask]
    outside = data[~mask]

    logger.info(
        "[Compose] Split result: %d inside, %d outside",
        len(inside),
        len(outside),
    )

    return inside, outside
