Learnable Modules
=================

PyTorch nn.Module classes for gradient-based optimization of processing parameters.

These modules enable learning optimal color grading, transforms, and filters
through backpropagation, useful for style transfer, scene optimization, and
neural rendering pipelines.

LearnableColor
--------------

.. autoclass:: gsmod.torch.learn.LearnableColor
   :members:
   :undoc-members:
   :show-inheritance:

LearnableTransform
------------------

.. autoclass:: gsmod.torch.learn.LearnableTransform
   :members:
   :undoc-members:
   :show-inheritance:

LearnableFilter
---------------

.. autoclass:: gsmod.torch.learn.LearnableFilter
   :members:
   :undoc-members:
   :show-inheritance:

LearnableGSTensor
-----------------

.. autoclass:: gsmod.torch.learn.LearnableGSTensor
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Classes
---------------------

.. autoclass:: gsmod.torch.learn.ColorGradingConfig
   :members:
   :undoc-members:

.. autoclass:: gsmod.torch.learn.TransformConfig
   :members:
   :undoc-members:

.. autoclass:: gsmod.torch.learn.LearnableFilterConfig
   :members:
   :undoc-members:

Usage Example
-------------

.. code-block:: python

   from gsmod.torch import GSTensorPro, LearnableColor
   from gsmod import ColorValues
   import torch

   # Create learnable color module from initial values
   initial = ColorValues(brightness=1.0, contrast=1.0, saturation=1.0)
   learnable = initial.learn("brightness", "saturation")

   # Or create directly
   learnable = LearnableColor(
       brightness=True,
       contrast=False,
       saturation=True
   )

   # Use in training loop
   optimizer = torch.optim.Adam(learnable.parameters(), lr=0.01)

   for epoch in range(100):
       # Apply learnable color to data
       result = learnable(data.sh0)

       # Compute loss
       loss = compute_loss(result, target)

       # Backpropagate
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

   # Extract learned values
   learned_values = learnable.to_values()
   print(f"Learned brightness: {learned_values.brightness}")
