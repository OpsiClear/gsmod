API Reference
=============

Complete API documentation for all public functions, classes, and modules.
All public symbols are re-exported through the top-level ``gsmod`` package,
so you can ``import gsmod`` and access everything from a single namespace.

Primary API
-----------

GSDataPro (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   gsdata_pro

Classes:

- :class:`gsmod.GSDataPro` - CPU data class with processing methods

Config Values
~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   config_values

Classes:

- :class:`gsmod.ColorValues` - Color adjustment parameters
- :class:`gsmod.FilterValues` - Filter parameters
- :class:`gsmod.TransformValues` - Transform parameters
- :class:`gsmod.OpacityValues` - Opacity adjustment parameters
- :class:`gsmod.HistogramConfig` - Histogram configuration

Presets
~~~~~~~

.. toctree::
   :maxdepth: 1

   presets

Color presets:

- ``WARM``, ``COOL``, ``NEUTRAL``, ``CINEMATIC``, ``VIBRANT``, ``MUTED``
- ``DRAMATIC``, ``VINTAGE``, ``GOLDEN_HOUR``, ``MOONLIGHT``

Filter presets:

- ``STRICT_FILTER``, ``QUALITY_FILTER``, ``CLEANUP_FILTER``

Opacity presets:

- ``FADE_MILD``, ``FADE_MODERATE``, ``BOOST_MILD``, ``BOOST_MODERATE``
- ``GHOST_EFFECT``, ``TRANSLUCENT``

Transform presets:

- ``DOUBLE_SIZE``, ``HALF_SIZE``, ``FLIP_X``, ``FLIP_Y``, ``FLIP_Z``

Loading functions:

- :func:`gsmod.get_color_preset` - Get color preset by name
- :func:`gsmod.get_filter_preset` - Get filter preset by name
- :func:`gsmod.get_transform_preset` - Get transform preset by name
- :func:`gsmod.get_opacity_preset` - Get opacity preset by name
- :func:`gsmod.load_color_json` - Load color values from JSON file
- :func:`gsmod.load_filter_json` - Load filter values from JSON file
- :func:`gsmod.load_transform_json` - Load transform values from JSON file
- :func:`gsmod.color_from_dict` - Create ColorValues from dict
- :func:`gsmod.filter_from_dict` - Create FilterValues from dict
- :func:`gsmod.transform_from_dict` - Create TransformValues from dict

Advanced Pipeline API
---------------------

Color Pipeline
~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   color

Classes:

- :class:`gsmod.Color` - Method-chaining color pipeline with LUT optimization
- :class:`gsmod.ColorPreset` - Color preset definition

Transform Pipeline
~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   transform

Classes:

- :class:`gsmod.Transform` - Method-chaining transform pipeline with matrix fusion

Quaternion utilities:

- :func:`gsmod.quaternion_multiply` - Multiply quaternions
- :func:`gsmod.quaternion_to_rotation_matrix` - Convert to rotation matrix
- :func:`gsmod.rotation_matrix_to_quaternion` - Convert from rotation matrix
- :func:`gsmod.axis_angle_to_quaternion` - Convert axis-angle to quaternion
- :func:`gsmod.euler_to_quaternion` - Convert Euler angles to quaternion
- :func:`gsmod.quaternion_to_euler` - Convert quaternion to Euler angles

Parameterization
~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   params

Classes:

- :class:`gsmod.Param` - Parameter specification for templates

GPU Acceleration
----------------

PyTorch Integration
~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   torch

GPU Data Container:

- :class:`gsmod.torch.GSTensorPro` - GPU tensor wrapper with same API as GSDataPro

GPU Pipeline Classes:

- :class:`gsmod.torch.ColorGPU` - GPU color pipeline
- :class:`gsmod.torch.TransformGPU` - GPU transform pipeline
- :class:`gsmod.torch.FilterGPU` - GPU filter pipeline
- :class:`gsmod.torch.PipelineGPU` - Unified GPU pipeline

Learnable Modules (Training)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   learnable

nn.Module classes for gradient-based optimization:

- :class:`gsmod.torch.LearnableColor` - Learnable color parameters
- :class:`gsmod.torch.LearnableTransform` - Learnable transform parameters
- :class:`gsmod.torch.LearnableFilter` - Learnable filter parameters
- :class:`gsmod.torch.LearnableGSTensor` - Full learnable GSTensor wrapper

Utilities
---------

Scene Composition
~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   compose

Functions:

- :func:`gsmod.concatenate` - Concatenate GSData objects
- :func:`gsmod.compose_with_transforms` - Compose scenes with transforms
- :func:`gsmod.deduplicate` - Remove duplicate Gaussians
- :func:`gsmod.merge_scenes` - Merge multiple scenes
- :func:`gsmod.split_by_region` - Split scene by spatial region

Filtering Utilities
~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   filter_utils

Functions:

- :func:`gsmod.calculate_scene_bounds` - Calculate scene bounding box
- :func:`gsmod.calculate_recommended_max_scale` - Get recommended max scale

Classes:

- :class:`gsmod.SceneBounds` - Scene bounds container

Histogram
~~~~~~~~~

.. toctree::
   :maxdepth: 1

   histogram

Classes:

- :class:`gsmod.HistogramResult` - Histogram computation result

Utility Functions
~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   utils

Functions:

- :func:`gsmod.linear_interp_1d` - Fast 1D linear interpolation
- :func:`gsmod.nearest_neighbor_1d` - Nearest neighbor lookup
- :func:`gsmod.multiply_opacity` - Multiply opacity values

Verification
~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   verification

Classes:

- :class:`gsmod.FormatVerifier` - Verify data format consistency

Protocols
~~~~~~~~~

- :class:`gsmod.PipelineStage` - Protocol for pipeline stages

Quick Reference
---------------

**Primary API**

- :class:`gsmod.GSDataPro` - Main CPU data class with ``.color()``, ``.filter()``, ``.transform()``
- :class:`gsmod.ColorValues` - Color parameters
- :class:`gsmod.FilterValues` - Filter parameters
- :class:`gsmod.TransformValues` - Transform parameters
- :class:`gsmod.HistogramConfig` - Histogram configuration

**GPU API**

- :class:`gsmod.torch.GSTensorPro` - GPU data class with same API
- :class:`gsmod.torch.PipelineGPU` - Unified GPU pipeline

**Advanced Pipelines**

- :class:`gsmod.Color` - Method-chaining color pipeline
- :class:`gsmod.Transform` - Method-chaining transform pipeline
- :class:`gsmod.Param` - Parameter specification for templates

**Learnable Modules**

- :class:`gsmod.torch.LearnableColor` - Learnable color grading
- :class:`gsmod.torch.LearnableTransform` - Learnable transforms
- :class:`gsmod.torch.LearnableFilter` - Learnable filtering

**Scene Operations**

- :func:`gsmod.concatenate` - Concatenate scenes
- :func:`gsmod.compose_with_transforms` - Compose with transforms
- :func:`gsmod.deduplicate` - Remove duplicates
- :func:`gsmod.calculate_scene_bounds` - Get scene bounds

**Quaternion Utilities**

- :func:`gsmod.quaternion_multiply` - Multiply quaternions
- :func:`gsmod.euler_to_quaternion` - Euler to quaternion
- :func:`gsmod.quaternion_to_rotation_matrix` - Quaternion to matrix

**Presets**

- Color: ``WARM``, ``COOL``, ``CINEMATIC``, ``VIBRANT``, ``DRAMATIC``, ``VINTAGE``
- Filter: ``STRICT_FILTER``, ``QUALITY_FILTER``, ``CLEANUP_FILTER``
- Transform: ``DOUBLE_SIZE``, ``HALF_SIZE``, ``FLIP_X``, ``FLIP_Y``, ``FLIP_Z``
