GPU Acceleration (PyTorch)
==========================

GPU-accelerated operations using PyTorch with up to 183x speedup.

GSTensorPro
-----------

.. automodule:: gsmod.torch.gstensor_pro
   :members:
   :undoc-members:
   :show-inheritance:

GPU Pipeline Classes
--------------------

ColorGPU
~~~~~~~~

.. autoclass:: gsmod.torch.color.ColorGPU
   :members:
   :undoc-members:
   :show-inheritance:

FilterGPU
~~~~~~~~~

.. autoclass:: gsmod.torch.filter.FilterGPU
   :members:
   :undoc-members:
   :show-inheritance:

TransformGPU
~~~~~~~~~~~~

.. autoclass:: gsmod.torch.transform.TransformGPU
   :members:
   :undoc-members:
   :show-inheritance:

PipelineGPU
~~~~~~~~~~~

.. autoclass:: gsmod.torch.pipeline.PipelineGPU
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

.. code-block:: python

   from gsmod.torch import GSTensorPro
   from gsmod import ColorValues, FilterValues, TransformValues

   # Load directly to GPU
   data = GSTensorPro.from_ply("scene.ply", device="cuda")

   # Same API as CPU
   data.filter(FilterValues(min_opacity=0.1, sphere_radius=5.0))
   data.transform(TransformValues.from_translation(1, 0, 0))
   data.color(ColorValues(brightness=1.2, saturation=1.3))

   # Save result
   data.to_ply("output.ply")

Performance
-----------

GPU benchmarks (1M Gaussians, RTX 3090 Ti):

- Peak speedup: 183.9x (sphere filter)
- Average speedup: 43.2x across all operations
- Throughput: 1.09 billion Gaussians/sec
