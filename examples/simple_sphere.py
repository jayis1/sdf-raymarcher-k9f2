#!/usr/bin/env python3
"""Example: Simple sphere scene rendered programmatically.

This demonstrates the simplest usage of the ray marcher library.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from raymarcher_k9f2 import (
    Scene, Camera, Renderer, RayMarcher, Material, Vec3,
    sdf_sphere, sdf_plane,
)

# Create a scene with a sphere on a checkerboard ground
scene = Scene()

# Red sphere
scene.add_object(
    lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0),
    Material(albedo=Vec3(0.9, 0.2, 0.15), specular=0.5, roughness=0.3)
)

# Checkerboard ground
scene.add_object(
    lambda p: sdf_plane(p, height=0.0),
    Material(
        albedo=Vec3(0.7, 0.7, 0.7), roughness=0.7,
        checker_scale=1.0, checker_color=Vec3(0.3, 0.3, 0.35)
    )
)

# Light
scene.add_directional_light(Vec3(0.5, 0.8, 0.3), Vec3(1.0, 0.95, 0.85), 1.0)

# Set up renderer
marcher = RayMarcher(enable_shadows=True)
renderer = Renderer(width=320, height=240, marcher=marcher)
camera = Camera(Vec3(5, 3, 5), Vec3(0, 1, 0), fov=55)

# Render and save
print("Rendering simple sphere scene...")
img = renderer.render(scene, camera)
renderer.save(img, 'example_simple.png')
print("Saved to example_simple.png")