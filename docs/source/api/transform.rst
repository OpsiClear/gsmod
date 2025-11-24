Transform Pipeline
==================

Geometric transformation pipeline with fused matrix operations.

.. automodule:: gsmod.transform.pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Quaternion Utilities
--------------------

.. automodule:: gsmod.transform.api
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

.. code-block:: python

   from gsmod import Transform

   # Method chaining with matrix fusion
   pipeline = (Transform()
       .translate([1, 0, 0])
       .rotate_euler(0, 45, 0)
       .scale(2.0))

   # Apply to data
   result = pipeline(data, inplace=True)

   # Use quaternion utilities
   from gsmod import quaternion_multiply, euler_to_quaternion

   q1 = euler_to_quaternion(0, 45, 0)
   q2 = euler_to_quaternion(0, 0, 90)
   q_combined = quaternion_multiply(q1, q2)
