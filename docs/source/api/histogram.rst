Histogram
=========

Histogram computation for Gaussian Splatting data analysis.

.. autoclass:: gsmod.histogram.HistogramResult
   :members:
   :no-index:

Usage Example
-------------

.. code-block:: python

   from gsmod import GSDataPro, HistogramConfig

   data = GSDataPro.from_ply("scene.ply")

   # Compute histogram with custom config
   config = HistogramConfig(num_bins=64, percentile_clip=99.0)
   result = data.compute_histogram(config)

   # Access results
   print(f"Luminance histogram: {result.luminance}")
   print(f"Per-channel histograms: {result.r}, {result.g}, {result.b}")
