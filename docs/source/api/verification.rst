Verification
============

Format verification utilities for ensuring data consistency.

.. automodule:: gsmod.verification
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

.. code-block:: python

   from gsmod import GSDataPro, FormatVerifier

   data = GSDataPro.from_ply("scene.ply")

   # Create verifier
   verifier = FormatVerifier(data)

   # Check format consistency
   is_valid = verifier.verify_all()
   if not is_valid:
       print("Format issues found:", verifier.issues)
