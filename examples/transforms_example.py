#!/usr/bin/env python3
"""Example: Using SDF transforms to create positioned and rotated objects.

Demonstrates the transform module for translate, rotate, and scale operations.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from raymarcher_k9f2 import (
    Scene, Camera, Renderer, RayMarcher, Material, Vec3,
    sdf_sphere, sdf_plane, sdf_torus,
)
from raymarcher_k9f2.transforms import sdf_translate, sdf_rotate_y, sdf_scale
from raymarcher_k9f2.material import gold, glass, checkerboard

# Create scene
scene = Scene()

# Ground plane
scene.add_object(
    lambda p: sdf_plane(p, height=0.0),
    checkerboard(scale=2.0)
)

# Gold sphere (translated to position)
scene.add_object(
    lambda p: sdf_translate(p, Vec3(-3, 1, 0), lambda q: sdf_sphere(q, Vec3(0, 0, 0), 1.0)),
    gold()
)

# Glass sphere
scene.add_object(
    lambda p: sdf_translate(p, Vec3(0, 1.2, 0), lambda q: sdf_sphere(q, Vec3(0, 0, 0), 1.2)),
    glass()
)

# Rotated torus
scene.add_object(
    lambda p: sdf_translate(
        p, Vec3(3, 1.0, 0),
        lambda q: sdf_rotate_y(q, 0.7854, lambda r: sdf_torus(r, Vec3(0, 0, 0), 0.8, 0.25))
    ),
    Material(albedo=Vec3(0.9, 0.85, 0.1), specular=0.7, roughness=0.15)
)

# Lights
scene.add_directional_light(Vec3(0.5, 0.8, 0.3), Vec3(1.0, 0.95, 0.85), 1.2)
scene.add_directional_light(Vec3(-0.3, 0.4, -0.5), Vec3(0.3, 0.4, 0.6), 0.3)
scene.add_point_light(Vec3(-2, 4, 2), Vec3(1.0, 0.7, 0.3), 4.0)

# Set up renderer
marcher = RayMarcher(enable_shadows=True, max_bounces=4, max_refraction_bounces=6)
renderer = Renderer(width=400, height=300, samples=1, marcher=marcher)
camera = Camera(Vec3(5, 3.5, 8), Vec3(0, 1, 0), fov=50)

print("Rendering transform example scene...")
img = renderer.render(scene, camera)
renderer.save(img, 'example_transforms.png')
print("Saved to example_transforms.png")