Scene Composition
=================

Utilities for combining and manipulating multiple Gaussian Splatting scenes.

.. automodule:: gsmod.compose
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

.. code-block:: python

   from gsmod import (
       GSDataPro,
       concatenate,
       compose_with_transforms,
       merge_scenes,
       deduplicate,
       split_by_region
   )
   from gsmod import TransformValues

   # Load multiple scenes
   scene1 = GSDataPro.from_ply("scene1.ply")
   scene2 = GSDataPro.from_ply("scene2.ply")

   # Simple concatenation
   combined = concatenate([scene1, scene2])

   # Compose with transforms
   transforms = [
       TransformValues.from_translation(-1, 0, 0),
       TransformValues.from_translation(1, 0, 0),
   ]
   composed = compose_with_transforms([scene1, scene2], transforms)

   # Remove duplicates
   cleaned = deduplicate(combined, threshold=0.01)

   # Split by region
   left, right = split_by_region(combined, axis=0, threshold=0.0)
