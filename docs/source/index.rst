Welcome to gsmod
================

**Ultra-fast CPU Processing for 3D Gaussian Splatting**

gsmod is a high-performance Python library for processing 3D Gaussian Splatting data.
Built with NumPy and Numba, it provides ultra-fast color adjustments, geometric transformations,
and spatial filtering with performance up to **1,091M Gaussians/sec** for color operations
and **698M Gaussians/sec** for transforms.

.. raw:: html

   <div class="admonition note" style="margin-top: 1rem; margin-bottom: 2rem;">
   <p class="admonition-title">Quick Start</p>
   <p>
   <code>pip install gsmod</code> |
   <code>from gsmod import GSDataPro, ColorValues</code> |
   <code>data.color(ColorValues(brightness=1.2))</code>
   </p>
   </div>

Key Features
------------

* **Ultra-fast color processing** - LUT-based color adjustments at 1,091M Gaussians/sec
* **Geometric transforms** - Translate, rotate, scale with fused matrix operations
* **Spatial filtering** - Sphere, box, opacity, and scale filters with boolean operators
* **GPU acceleration** - Optional PyTorch backend with up to 183x speedup
* **Config values** - Declarative, serializable parameter objects
* **Presets** - Built-in color, filter, and transform presets
* **Zero-copy processing** - In-place operations for memory efficiency

Performance Highlights
-----------------------

* **Color**: 1,091M Gaussians/sec (0.092ms for 100K Gaussians)
* **Transform**: 698M Gaussians/sec (1.43ms for 1M Gaussians)
* **Filter**: 46M Gaussians/sec (2.2ms for 100K Gaussians)
* **GPU**: Up to 183x speedup, 1.09B Gaussians/sec on RTX 3090 Ti

Documentation
-------------

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   overview
   usage
   changelog

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/index

Additional Resources
--------------------

* :ref:`genindex` - Complete function and class index
* :ref:`modindex` - Module index
* :ref:`search` - Full-text search

.. raw:: html

   <hr style="margin-top: 3rem; border: none; border-top: 1px solid #e0e0e0;">
   <p style="text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;">
   Documentation built with <a href="https://www.sphinx-doc.org/">Sphinx</a> and
   <a href="https://pradyunsg.me/furo/">Furo</a> theme.
   </p>
