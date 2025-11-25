Presets
=======

Built-in preset configurations for common operations.

.. automodule:: gsmod.config.presets
   :members:
   :undoc-members:
   :show-inheritance:

Color Presets
-------------

- ``WARM`` - Warm orange tones
- ``COOL`` - Cool blue tones
- ``NEUTRAL`` - Balanced neutral look
- ``CINEMATIC`` - Film-like cinematic look
- ``VIBRANT`` - High saturation vibrant colors
- ``MUTED`` - Desaturated muted tones
- ``DRAMATIC`` - High contrast dramatic look
- ``VINTAGE`` - Faded vintage film look
- ``GOLDEN_HOUR`` - Golden sunset tones
- ``MOONLIGHT`` - Cool moonlit night look

Filter Presets
--------------

- ``STRICT_FILTER`` - Aggressive filtering for clean results
- ``QUALITY_FILTER`` - Balanced quality filtering
- ``CLEANUP_FILTER`` - Light cleanup filtering

Opacity Presets
---------------

- ``FADE_MILD`` - Reduce opacity to 90%
- ``FADE_MODERATE`` - Reduce opacity to 70%
- ``BOOST_MILD`` - Increase opacity by 10%
- ``BOOST_MODERATE`` - Increase opacity by 30%
- ``GHOST_EFFECT`` - Semi-transparent ghost effect (50% opacity)
- ``TRANSLUCENT`` - Highly transparent (30% opacity)

Transform Presets
-----------------

- ``DOUBLE_SIZE`` - Scale up 2x
- ``HALF_SIZE`` - Scale down 0.5x
- ``FLIP_X`` - Mirror along X axis
- ``FLIP_Y`` - Mirror along Y axis
- ``FLIP_Z`` - Mirror along Z axis

Usage Examples
--------------

.. code-block:: python

   from gsmod import GSDataPro, CINEMATIC, STRICT_FILTER, DOUBLE_SIZE
   from gsmod import ColorValues, FADE_MODERATE, GHOST_EFFECT

   data = GSDataPro.from_ply("scene.ply")

   # Apply presets
   data.color(CINEMATIC)
   data.filter(STRICT_FILTER)
   data.transform(DOUBLE_SIZE)
   data.opacity(FADE_MODERATE)  # Fade to 70% opacity

   # Compose presets with custom values
   data.color(CINEMATIC + ColorValues(brightness=1.1))

   # Create ghost effect
   data.opacity(GHOST_EFFECT)  # 50% opacity

   # Load from JSON
   from gsmod import load_color_json
   custom = load_color_json("my_preset.json")
   data.color(custom)
