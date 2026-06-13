"""
SDF Ray Marcher k9f2 — A pure-Python signed distance field ray marcher.

Renders 3D scenes using ray marching against analytic signed distance fields.
Supports multiple primitives, materials, soft shadows, ambient occlusion,
reflections, refractions (glass), subsurface scattering, depth-of-field,
animation, and YAML/TOML scene configuration.

Usage::

    from raymarcher_k9f2 import Scene, Camera, Renderer, RayMarcher, Material, Vec3
    from raymarcher_k9f2.primitives import sdf_sphere, sdf_plane

    scene = Scene()
    scene.add_object(lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0),
                     Material(albedo=Vec3(1, 0, 0)))
    scene.add_directional_light(Vec3(0.5, 0.8, 0.3))

    marcher = RayMarcher(enable_shadows=True)
    renderer = Renderer(width=800, height=600, marcher=marcher)
    camera = Camera(Vec3(5, 3, 5), Vec3(0, 1, 0), fov=55)
    img = renderer.render(scene, camera)
    renderer.save(img, "output.png")

CLI::

    raymarcher-k9f2 --scene demo --width 800 --height 600 --output output.png
    raymarcher-k9f2 --config scene.yaml --output custom.png
"""

__version__ = "1.1.0"
__author__ = "SDF Ray Marcher Project"

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

__all__ = [
    "Vec3", "BLACK", "WHITE", "SKY_BLUE",
    "Material",
    "sdf_sphere", "sdf_box", "sdf_torus", "sdf_cylinder", "sdf_capsule",
    "sdf_plane", "sdf_cone", "sdf_hex_prism", "sdf_mandelbulb",
    "sdf_union", "sdf_smooth_union", "sdf_smooth_subtraction",
    "sdf_subtraction", "sdf_intersection",
    "checkerboard",
    "Scene", "SceneObject", "PointLight", "DirectionalLight", "HitResult",
    "RayMarcher",
    "Camera",
    "Renderer",
    "build_demo_scene", "build_chess_scene", "build_terrain_scene",
    "build_abstract_scene", "build_glass_scene", "build_fractal_scene",
    "SCENES",
]