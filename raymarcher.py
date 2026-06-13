#!/usr/bin/env python3
"""
Backward-compatible shim: imports everything from the raymarcher_k9f2 package.

This file preserves the original single-file interface. New code should use:
    from raymarcher_k9f2 import Vec3, Scene, ...
"""

import sys
import os

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from raymarcher_k9f2 import *  # noqa: F401, F403
from raymarcher_k9f2 import __version__
from raymarcher_k9f2.vec3 import Vec3, BLACK, WHITE, SKY_BLUE
from raymarcher_k9f2.material import Material
from raymarcher_k9f2.primitives import (
    sdf_sphere, sdf_box, sdf_torus, sdf_cylinder, sdf_capsule,
    sdf_plane, sdf_cone, sdf_hex_prism, sdf_mandelbulb,
)
from raymarcher_k9f2.csg import (
    sdf_union, sdf_smooth_union, sdf_smooth_subtraction,
    sdf_subtraction, sdf_intersection,
)
from raymarcher_k9f2.patterns import checkerboard
from raymarcher_k9f2.scene import Scene, SceneObject, PointLight, DirectionalLight, HitResult
from raymarcher_k9f2.marcher import RayMarcher
from raymarcher_k9f2.camera import Camera
from raymarcher_k9f2.renderer import Renderer
from raymarcher_k9f2.scenes import (
    build_demo_scene, build_chess_scene, build_terrain_scene,
    build_abstract_scene, build_glass_scene, build_fractal_scene,
    SCENES,
)

# Re-export the original main() for CLI compatibility
from raymarcher_k9f2.cli import main

if __name__ == '__main__':
    main()