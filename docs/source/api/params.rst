Parameterization
================

Tools for creating parameterized pipeline templates with efficient caching.

Param
-----

.. autoclass:: gsmod.params.Param
   :members:
   :no-index:

Usage Example
-------------

.. code-block:: python

   from gsmod import Color, Param

   # Create template with named parameters
   template = Color.template(
       brightness=Param("b", default=1.2, range=(0.5, 2.0)),
       contrast=Param("c", default=1.1, range=(0.5, 2.0)),
       saturation=Param("s", default=1.3, range=(0.0, 3.0))
   )

   # Use with different parameters (auto-cached for performance)
   result1 = template(data, params={"b": 1.5, "c": 1.2, "s": 1.4})
   result2 = template(data, params={"b": 0.8, "c": 1.0, "s": 1.0})

   # Animation use case - cached LUTs for efficiency
   import numpy as np
   for t in np.linspace(0, 1, 100):
       brightness = 1.0 + t * 1.0  # Animate 1.0 to 2.0
       result = template(data, params={"b": brightness})
