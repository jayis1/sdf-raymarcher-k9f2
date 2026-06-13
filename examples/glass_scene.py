#!/usr/bin/env python3
"""Example: Glass sphere scene with refraction and depth-of-field.

Demonstrates glass materials, refraction, and the thin-lens DOF model.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from raymarcher_k9f2 import (
    Scene, Camera, Renderer, RayMarcher, Material, Vec3,
    sdf_sphere, sdf_plane,
)
from raymarcher_k9f2.material import glass

# Create scene
scene = Scene()

# Ground
scene.add_object(
    lambda p: sdf_plane(p, height=0.0),
    Material(
        albedo=Vec3(0.8, 0.8, 0.8), roughness=0.7, reflectivity=0.2,
        checker_scale=2.0, checker_color=Vec3(0.3, 0.3, 0.3)
    )
)

# Glass sphere (using preset)
glass_mat = glass(ior=1.5)
scene.add_object(lambda p: sdf_sphere(p, Vec3(0, 1.2, 0), 1.2), glass_mat)

# Red sphere for contrast
scene.add_object(
    lambda p: sdf_sphere(p, Vec3(2.5, 0.6, 1.5), 0.6),
    Material(albedo=Vec3(0.9, 0.1, 0.1), specular=0.7, roughness=0.15)
)

# Gold sphere
from raymarcher_k9f2.material import gold as gold_preset
scene.add_object(
    lambda p: sdf_sphere(p, Vec3(-2, 1, 1), 1.0),
    gold_preset()
)

# Lights
scene.add_directional_light(Vec3(0.5, 0.8, 0.3), Vec3(1.0, 0.95, 0.85), 1.2)
scene.add_point_light(Vec3(-3, 4, 2), Vec3(1.0, 0.7, 0.3), 4.0)

# Set up renderer with depth of field
marcher = RayMarcher(enable_shadows=True, max_bounces=4, max_refraction_bounces=6)
renderer = Renderer(width=400, height=300, samples=1, marcher=marcher)
camera = Camera(Vec3(5, 3.5, 8), Vec3(0, 1, 0), fov=50, focal_distance=8.0, aperture=0.12)

print("Rendering glass scene with DOF...")
img = renderer.render(scene, camera)
renderer.save(img, 'example_glass.png')
print("Saved to example_glass.png")