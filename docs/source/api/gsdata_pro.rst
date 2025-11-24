GSDataPro
=========

The primary API for processing Gaussian Splatting data on CPU.

.. automodule:: gsmod.gsdata_pro
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

.. code-block:: python

   from gsmod import GSDataPro, ColorValues, FilterValues, TransformValues

   # Load data
   data = GSDataPro.from_ply("scene.ply")

   # Apply operations
   data.color(ColorValues(brightness=1.2, saturation=1.3))
   data.filter(FilterValues(min_opacity=0.1, sphere_radius=5.0))
   data.transform(TransformValues.from_translation(1, 0, 0))

   # Save result
   data.to_ply("output.ply")
