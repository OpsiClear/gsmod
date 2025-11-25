Config Values
=============

Declarative parameter objects for color, filter, and transform operations.

ColorValues
-----------

.. autoclass:: gsmod.config.values.ColorValues
   :members:
   :undoc-members:
   :show-inheritance:

FilterValues
------------

.. autoclass:: gsmod.config.values.FilterValues
   :members:
   :undoc-members:
   :show-inheritance:

TransformValues
---------------

.. autoclass:: gsmod.config.values.TransformValues
   :members:
   :undoc-members:
   :show-inheritance:

OpacityValues
-------------

.. autoclass:: gsmod.config.values.OpacityValues
   :members:
   :undoc-members:
   :show-inheritance:

HistogramConfig
---------------

.. autoclass:: gsmod.config.values.HistogramConfig
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

.. code-block:: python

   from gsmod import ColorValues, FilterValues, TransformValues

   # Create color values
   color = ColorValues(brightness=1.2, contrast=1.1, saturation=1.3)

   # Compose with presets
   from gsmod import CINEMATIC
   custom = CINEMATIC + ColorValues(brightness=1.1)

   # Create filter values
   filter_vals = FilterValues(min_opacity=0.1, max_scale=2.5, sphere_radius=5.0)

   # Create transform values with factory methods
   translation = TransformValues.from_translation(1, 0, 0)
   rotation = TransformValues.from_rotation_euler(0, 45, 0)
   scale = TransformValues.from_scale(2.0)
