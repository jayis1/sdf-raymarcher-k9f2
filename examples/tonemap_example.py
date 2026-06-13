#!/usr/bin/env python3
"""Example: Tone mapping with ACES filmic curve.

Demonstrates the tone mapping module for better highlight handling.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from raymarcher_k9f2 import (
    Scene, Camera, Renderer, RayMarcher, Material, Vec3,
    sdf_sphere, sdf_plane,
)
from raymarcher_k9f2.material import gold, glass
from raymarcher_k9f2.tonemap import apply_tonemap
import numpy as np

# Create a high-contrast scene
scene = Scene()

# Ground
scene.add_object(
    lambda p: sdf_plane(p, height=0.0),
    Material(albedo=Vec3(0.7, 0.7, 0.7), roughness=0.7, checker_scale=2.0, checker_color=Vec3(0.3, 0.3, 0.3))
)

# Gold sphere (reflective, will have bright highlights)
scene.add_object(lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0), gold())

# Glass sphere (refractive)
scene.add_object(lambda p: sdf_sphere(p, Vec3(3, 1, 1), 1.0), glass())

# Emissive sphere (very bright, will benefit from tone mapping)
scene.add_object(
    lambda p: sdf_sphere(p, Vec3(-3, 0.5, 0), 0.5),
    Material(albedo=Vec3(0, 0, 0), emissive=Vec3(3, 2.5, 1))
)

# Strong lighting
scene.add_directional_light(Vec3(0.5, 0.8, 0.3), Vec3(1.2, 1.1, 0.9), 1.5)
scene.add_point_light(Vec3(0, 5, 0), Vec3(2, 1.8, 1.5), 8.0)

# Render with low gamma to get HDR data
marcher = RayMarcher(enable_shadows=True, max_bounces=3)
renderer = Renderer(width=400, height=300, marcher=marcher, gamma=1.0, exposure=1.0)
camera = Camera(Vec3(6, 3, 6), Vec3(0, 1, 0), fov=55)

print("Rendering with tone mapping example...")
img = renderer.render(scene, camera)

# Save without tone mapping (raw)
renderer.save(img, 'example_no_tonemap.png')

# Apply ACES tone mapping
img_aces = apply_tonemap(img, method='aces', exposure=1.0)
renderer.save(img_aces, 'example_aces_tonemap.png')

# Apply Reinhard tone mapping
img_reinhard = apply_tonemap(img, method='reinhard', exposure=1.0)
renderer.save(img_reinhard, 'example_reinhard_tonemap.png')

print("Saved: example_no_tonemap.png, example_aces_tonemap.png, example_reinhard_tonemap.png")