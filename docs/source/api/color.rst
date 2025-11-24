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
