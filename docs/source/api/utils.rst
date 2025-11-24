Utility Functions
=================

Low-level utility functions for interpolation and array operations.

Interpolation
-------------

.. autofunction:: gsmod.utils.linear_interp_1d

.. autofunction:: gsmod.utils.nearest_neighbor_1d

Opacity Operations
------------------

.. autofunction:: gsmod.utils.multiply_opacity

Usage Example
-------------

.. code-block:: python

   from gsmod import linear_interp_1d, nearest_neighbor_1d, multiply_opacity
   import numpy as np

   # Linear interpolation for LUT application
   lut = np.linspace(0, 1, 256) ** 2.2  # Gamma curve
   values = np.random.rand(1000)
   result = linear_interp_1d(values, lut)

   # Nearest neighbor lookup
   result_nn = nearest_neighbor_1d(values, lut)

   # Multiply opacity
   data = GSDataPro.from_ply("scene.ply")
   multiply_opacity(data.opacities, 0.8, inplace=True)
