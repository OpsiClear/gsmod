Color Pipeline
==============

Advanced color processing pipeline with LUT optimization.

.. automodule:: gsmod.color.pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

.. code-block:: python

   from gsmod import Color

   # Method chaining with LUT optimization
   pipeline = (Color()
       .brightness(1.2)
       .contrast(1.1)
       .saturation(1.3)
       .temperature(0.1)
       .shadows(0.05)
       .highlights(-0.05))

   # Apply to data
   result = pipeline(data, inplace=True)

   # Reuse compiled LUT for multiple datasets
   result2 = pipeline(data2, inplace=True)


Auto-Correction
===============

Industry-standard automatic color correction algorithms inspired by
Photoshop, Lightroom, and iOS Photos.

.. automodule:: gsmod.color.auto
   :members:
   :undoc-members:
   :show-inheritance:

Auto-Correction Example
-----------------------

.. code-block:: python

   from gsmod import GSDataPro
   from gsmod.color import auto_enhance, auto_contrast, auto_white_balance

   data = GSDataPro.from_ply("scene.ply")

   # Quick auto-enhance (like iOS Photos Auto)
   result = auto_enhance(data, strength=0.8)
   data.color(result.to_color_values())

   # Or use individual corrections:

   # Auto Contrast - Photoshop style (0.1% percentile clipping)
   contrast_result = auto_contrast(data, clip_percent=0.1)

   # Auto White Balance - Gray World assumption
   wb_result = auto_white_balance(data, method="gray_world")
   print(f"Temperature: {wb_result.temperature}")

Available Functions
-------------------

- ``auto_enhance``: Combined enhancement (exposure + contrast + white balance)
- ``auto_contrast``: Percentile-based histogram stretching
- ``auto_exposure``: 18% gray midtone targeting
- ``auto_white_balance``: Gray World or White Patch methods
- ``compute_optimal_parameters``: Minimal adjustments to reach targets
