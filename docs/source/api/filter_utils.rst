Filter Utilities
================

Utilities for spatial filtering and scene bounds calculation.

Scene Bounds
------------

.. autofunction:: gsmod.filter.bounds.calculate_scene_bounds

.. autofunction:: gsmod.filter.bounds.calculate_recommended_max_scale

.. autoclass:: gsmod.filter.bounds.SceneBounds
   :members:
   :no-index:

Usage Example
-------------

.. code-block:: python

   from gsmod import (
       GSDataPro,
       calculate_scene_bounds,
       calculate_recommended_max_scale,
       SceneBounds
   )

   data = GSDataPro.from_ply("scene.ply")

   # Calculate scene bounds
   bounds = calculate_scene_bounds(data.means)
   print(f"Center: {bounds.center}")
   print(f"Size: {bounds.size}")
   print(f"Diagonal: {bounds.diagonal}")

   # Get recommended max scale
   max_scale = calculate_recommended_max_scale(data.scales, percentile=99)
   print(f"Recommended max scale: {max_scale}")
