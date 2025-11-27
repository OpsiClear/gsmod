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


Loss Functions (PyTorch)
========================

Differentiable loss functions for histogram-based learning.
These are used to optimize color parameters via gradient descent.

Basic Loss
----------

.. autoclass:: gsmod.histogram.MomentMatchingLoss
   :members:
   :no-index:

Perceptual Loss Functions
-------------------------

Advanced loss functions that preserve visual quality while matching targets.

.. autoclass:: gsmod.histogram.PerceptualColorLoss
   :members:
   :no-index:

.. autoclass:: gsmod.histogram.ContrastPreservationLoss
   :members:
   :no-index:

.. autoclass:: gsmod.histogram.ParameterBoundsLoss
   :members:
   :no-index:

Factory Function
----------------

.. autofunction:: gsmod.histogram.create_balanced_loss
   :no-index:

Soft Histogram Functions
------------------------

.. autofunction:: gsmod.histogram.soft_histogram
   :no-index:

.. autofunction:: gsmod.histogram.soft_histogram_rgb
   :no-index:

Loss Functions Example
----------------------

.. code-block:: python

   import torch
   from gsmod.histogram import (
       PerceptualColorLoss,
       ContrastPreservationLoss,
       create_balanced_loss,
   )

   # Create perceptual loss (addresses flat histogram problem)
   loss_fn = PerceptualColorLoss(
       moment_weight=1.0,
       contrast_weight=0.5,      # Preserve contrast
       dynamic_range_weight=0.3, # Match percentiles
       regularization_weight=0.1,
       min_contrast_ratio=0.7,   # Don't reduce below 70%
   )

   # Or use the factory function for balanced defaults
   loss_fn = create_balanced_loss(
       target_hist,
       contrast_weight=0.5,
       regularization_weight=0.1,
   )

   # Use in training loop
   adjusted = learnable_color(colors)
   loss = loss_fn(adjusted, original_colors, target_hist, learnable_color)

Why Perceptual Loss?
--------------------

The basic ``MomentMatchingLoss`` only matches mean and std, which can
be achieved by reducing contrast and shifting brightness - resulting
in flat, washed-out images.

``PerceptualColorLoss`` addresses this by:

1. **Contrast preservation**: Penalizes contrast reduction below threshold
2. **Dynamic range matching**: Uses 5th/95th percentiles
3. **Parameter regularization**: Keeps adjustments near neutral values

This produces visually pleasing results similar to professional photo
editing software.
