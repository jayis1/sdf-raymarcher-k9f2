"""Built-in scene builders for various demo scenes."""

import math

from raymarcher_k9f2.vec3 import Vec3
from raymarcher_k9f2.material import Material
from raymarcher_k9f2.primitives import (
    sdf_sphere, sdf_box, sdf_torus, sdf_cylinder, sdf_capsule,
    sdf_plane, sdf_cone, sdf_mandelbulb,
)
from raymarcher_k9f2.csg import sdf_smooth_union, sdf_smooth_subtraction
from raymarcher_k9f2.scene import Scene


def build_demo_scene(time_val: float = 0.0) -> Scene:
    """Default demo scene with various primitives."""
    scene = Scene()
    scene.time = time_val

    ground_mat = Material(
        albedo=Vec3(0.7, 0.7, 0.7),
        specular=0.2, roughness=0.7, reflectivity=0.15,
        checker_scale=1.0, checker_color=Vec3(0.3, 0.3, 0.35),
    )
    scene.add_object(lambda p: sdf_plane(p, height=0.0), ground_mat)

    # Animated sphere (red, reflective)
    def sphere_sdf(p: Vec3) -> float:
        center = Vec3(0, 1.0 + 0.3 * math.sin(time_val * 1.5), 0)
        return sdf_sphere(p, center, 1.0)

    sphere_mat = Material(
        albedo=Vec3(0.9, 0.2, 0.15),
        specular=0.9, roughness=0.05, reflectivity=0.4, fresnel=0.05,
    )
    scene.add_object(sphere_sdf, sphere_mat)

    # Box (blue)
    box_mat = Material(albedo=Vec3(0.2, 0.5, 0.9), specular=0.6, roughness=0.2, reflectivity=0.1)
    scene.add_object(lambda p: sdf_box(p, Vec3(3.0, 0.75, 1.0), Vec3(0.75, 0.75, 0.75)), box_mat)

    # Torus (yellow)
    torus_mat = Material(albedo=Vec3(0.9, 0.85, 0.1), specular=0.7, roughness=0.15)
    scene.add_object(lambda p: sdf_torus(p, Vec3(-2.5, 0.8, 2.0), 0.8, 0.25), torus_mat)

    # Capsule (green)
    capsule_mat = Material(albedo=Vec3(0.1, 0.8, 0.4), specular=0.5, roughness=0.3)
    scene.add_object(lambda p: sdf_capsule(p, Vec3(-3.0, 0.5, -1.5), Vec3(-3.0, 2.5, -1.5), 0.35), capsule_mat)

    # Smooth union of two spheres (orange)
    def smooth_sdf(p: Vec3) -> float:
        d1 = sdf_sphere(p, Vec3(1.5, 0.7, -2.5), 0.7)
        d2 = sdf_sphere(p, Vec3(2.5, 0.7, -2.5), 0.7)
        return sdf_smooth_union(d1, d2, k=0.5)

    smooth_mat = Material(albedo=Vec3(0.85, 0.55, 0.15), specular=0.4, roughness=0.4)
    scene.add_object(smooth_sdf, smooth_mat)

    # Cylinder (purple)
    cyl_mat = Material(albedo=Vec3(0.7, 0.3, 0.8), specular=0.5, roughness=0.2)
    scene.add_object(lambda p: sdf_cylinder(p, Vec3(3.5, 0.6, -3.0), 0.5, 0.6), cyl_mat)

    scene.add_directional_light(Vec3(0.5, 0.8, 0.3), Vec3(1.0, 0.95, 0.85), 1.0)
    scene.add_directional_light(Vec3(-0.3, 0.4, -0.5), Vec3(0.3, 0.4, 0.6), 0.3)
    scene.add_point_light(Vec3(2, 3, -2), Vec3(1.0, 0.8, 0.5), 5.0)

    return scene


def build_chess_scene(time_val: float = 0.0) -> Scene:
    """Chess-like scene with a reflective board and pieces."""
    scene = Scene()
    scene.time = time_val
    scene.fog_density = 0.015
    scene.fog_color = Vec3(0.35, 0.3, 0.25)

    board_mat = Material(albedo=Vec3(0.15, 0.12, 0.1), specular=0.3, roughness=0.6, reflectivity=0.3)
    scene.add_object(lambda p: sdf_box(p, Vec3(0, -0.1, 0), Vec3(5, 0.1, 5)), board_mat)

    for ix in range(-4, 5):
        for iz in range(-4, 5):
            if (ix + iz) % 2 == 0:
                continue
            sq_mat = Material(albedo=Vec3(0.85, 0.82, 0.75), specular=0.5, roughness=0.4, reflectivity=0.1)

            def make_sq_sdf(cx, cz):
                def sdf(p: Vec3) -> float:
                    return sdf_box(p, Vec3(cx, 0.001, cz), Vec3(0.48, 0.001, 0.48))
                return sdf
            scene.add_object(make_sq_sdf(ix, iz), sq_mat)

    # King
    king_mat = Material(albedo=Vec3(0.95, 0.92, 0.85), specular=0.8, roughness=0.1, reflectivity=0.2)

    def king_sdf(p: Vec3) -> float:
        body = sdf_cylinder(p, Vec3(0, 0.7, 0), 0.25, 0.7)
        head = sdf_sphere(p, Vec3(0, 1.6, 0), 0.22)
        base = sdf_cylinder(p, Vec3(0, 0.15, 0), 0.35, 0.15)
        return sdf_smooth_union(sdf_smooth_union(body, head, k=0.15), base, k=0.1)

    scene.add_object(king_sdf, king_mat)

    # Pawn
    pawn_mat = Material(albedo=Vec3(0.12, 0.1, 0.08), specular=0.6, roughness=0.2)

    def pawn_sdf(p: Vec3) -> float:
        body = sdf_cylinder(p, Vec3(2, 0.35, 1), 0.15, 0.35)
        head = sdf_sphere(p, Vec3(2, 0.8, 1), 0.17)
        base = sdf_cylinder(p, Vec3(2, 0.08, 1), 0.22, 0.08)
        return sdf_smooth_union(sdf_smooth_union(body, head, k=0.08), base, k=0.05)

    scene.add_object(pawn_sdf, pawn_mat)

    scene.add_directional_light(Vec3(0.4, 0.9, 0.3), Vec3(1.0, 0.95, 0.85), 1.2)
    scene.add_point_light(Vec3(0, 6, 0), Vec3(0.9, 0.8, 0.6), 8.0)

    return scene


def build_terrain_scene(time_val: float = 0.0) -> Scene:
    """Procedural terrain scene with water."""
    scene = Scene()
    scene.time = time_val
    scene.fog_density = 0.02
    scene.fog_color = Vec3(0.6, 0.7, 0.8)

    def noise2d(x: float, z: float) -> float:
        return (math.sin(x * 0.5) * math.cos(z * 0.7) * 0.5 +
                math.sin(x * 1.3 + z * 0.9) * 0.25 +
                math.sin(x * 2.7 + z * 2.1) * 0.125)

    def terrain_sdf(p: Vec3) -> float:
        h = noise2d(p.x, p.z) + 1.0
        return p.y - h

    terrain_mat = Material(albedo=Vec3(0.35, 0.55, 0.25), specular=0.1, roughness=0.9)
    scene.add_object(terrain_sdf, terrain_mat)

    # Water plane with animation
    def water_sdf(p: Vec3) -> float:
        wave = math.sin(p.x * 2.0 + time_val) * 0.02 + math.cos(p.z * 3.0 + time_val * 0.7) * 0.015
        return sdf_plane(p, height=0.3 + wave)

    water_mat = Material(
        albedo=Vec3(0.15, 0.3, 0.6), specular=0.9, roughness=0.05,
        reflectivity=0.6, transparency=0.3, ior=1.33,
    )
    scene.add_object(water_sdf, water_mat)

    scene.add_directional_light(Vec3(0.3, 0.8, 0.5), Vec3(1.0, 0.9, 0.7), 1.0)
    scene.add_directional_light(Vec3(-0.5, 0.3, -0.3), Vec3(0.4, 0.5, 0.7), 0.3)

    return scene


def build_abstract_scene(time_val: float = 0.0) -> Scene:
    """Abstract scene with morphing smooth unions."""
    scene = Scene()
    scene.time = time_val

    def abstract_sdf(p: Vec3) -> float:
        d = float('inf')
        for i in range(5):
            angle = time_val * 0.5 + i * math.pi * 2.0 / 5.0
            cx = 2.0 * math.cos(angle)
            cy = 1.5 + 0.5 * math.sin(time_val + i)
            cz = 2.0 * math.sin(angle)
            d = sdf_smooth_union(d, sdf_sphere(p, Vec3(cx, cy, cz), 0.8), k=1.2)
        return d

    abstract_mat = Material(
        albedo=Vec3(0.8, 0.3, 0.7), specular=0.8, roughness=0.1,
        reflectivity=0.5, fresnel=0.03,
    )
    scene.add_object(abstract_sdf, abstract_mat)

    ground_mat = Material(albedo=Vec3(0.2, 0.2, 0.25), specular=0.2, roughness=0.8, reflectivity=0.3)
    scene.add_object(lambda p: sdf_plane(p, height=0.0), ground_mat)

    scene.add_directional_light(Vec3(0.5, 0.8, 0.3), Vec3(1.0, 0.95, 0.85), 1.0)
    scene.add_directional_light(Vec3(-0.3, 0.5, -0.7), Vec3(0.5, 0.3, 0.7), 0.4)

    return scene


def build_glass_scene(time_val: float = 0.0) -> Scene:
    """Scene showcasing glass/refraction and subsurface scattering."""
    scene = Scene()
    scene.time = time_val

    ground_mat = Material(
        albedo=Vec3(0.8, 0.8, 0.8), specular=0.2, roughness=0.7,
        reflectivity=0.2, checker_scale=2.0, checker_color=Vec3(0.3, 0.3, 0.3),
    )
    scene.add_object(lambda p: sdf_plane(p, height=0.0), ground_mat)

    # Glass sphere (center)
    glass_mat = Material(
        albedo=Vec3(0.95, 0.95, 0.98), specular=0.9, roughness=0.02,
        reflectivity=0.3, fresnel=0.04, transparency=0.9, ior=1.5,
    )
    scene.add_object(lambda p: sdf_sphere(p, Vec3(0, 1.2, 0), 1.2), glass_mat)

    # Subsurface scattering sphere (wax/candle-like)
    sss_mat = Material(
        albedo=Vec3(0.9, 0.7, 0.5), specular=0.3, roughness=0.6,
        subsurface=0.8, subsurface_color=Vec3(0.9, 0.4, 0.1),
    )
    scene.add_object(lambda p: sdf_sphere(p, Vec3(-3, 0.8, 1), 0.8), sss_mat)

    # Glass cube
    glass_cube_mat = Material(
        albedo=Vec3(0.95, 0.95, 0.97), specular=0.8, roughness=0.02,
        reflectivity=0.2, fresnel=0.04, transparency=0.85, ior=1.45,
    )
    scene.add_object(lambda p: sdf_box(p, Vec3(3, 0.75, -0.5), Vec3(0.75, 0.75, 0.75)), glass_cube_mat)

    # Opaque red sphere
    red_mat = Material(albedo=Vec3(0.9, 0.1, 0.1), specular=0.7, roughness=0.15, reflectivity=0.3)
    scene.add_object(lambda p: sdf_sphere(p, Vec3(1.5, 0.6, 2.0), 0.6), red_mat)

    # Water sphere
    water_mat = Material(
        albedo=Vec3(0.7, 0.85, 0.95), specular=0.9, roughness=0.01,
        reflectivity=0.15, transparency=0.7, ior=1.33,
    )
    scene.add_object(lambda p: sdf_sphere(p, Vec3(-1.5, 0.7, -2.5), 0.7), water_mat)

    scene.add_directional_light(Vec3(0.5, 0.8, 0.3), Vec3(1.0, 0.95, 0.85), 1.2)
    scene.add_directional_light(Vec3(-0.3, 0.4, -0.5), Vec3(0.3, 0.4, 0.6), 0.3)
    scene.add_point_light(Vec3(-2, 3, 2), Vec3(1.0, 0.7, 0.3), 4.0)

    return scene


def build_fractal_scene(time_val: float = 0.0) -> Scene:
    """Mandelbulb fractal scene."""
    scene = Scene()
    scene.time = time_val
    scene.fog_density = 0.01
    scene.fog_color = Vec3(0.1, 0.1, 0.15)

    power = 8.0 + 2.0 * math.sin(time_val * 0.3)

    def fractal_sdf(p: Vec3) -> float:
        return sdf_mandelbulb(p, Vec3(0, 1.5, 0), scale=1.2, power=power, iterations=8)

    fractal_mat = Material(albedo=Vec3(0.8, 0.6, 0.3), specular=0.5, roughness=0.4, reflectivity=0.1)
    scene.add_object(fractal_sdf, fractal_mat)

    ground_mat = Material(albedo=Vec3(0.15, 0.15, 0.2), specular=0.3, roughness=0.8, reflectivity=0.3)
    scene.add_object(lambda p: sdf_plane(p, height=0.0), ground_mat)

    scene.add_directional_light(Vec3(0.5, 0.8, 0.3), Vec3(1.0, 0.95, 0.85), 1.0)
    scene.add_directional_light(Vec3(-0.5, 0.3, -0.5), Vec3(0.4, 0.3, 0.6), 0.4)
    scene.add_point_light(Vec3(3, 4, -2), Vec3(0.8, 0.6, 1.0), 5.0)

    return scene


def build_studio_scene(time_val: float = 0.0) -> Scene:
    """Studio scene with three-point lighting and a variety of materials."""
    scene = Scene()
    scene.time = time_val

    # Ground plane
    ground_mat = Material(
        albedo=Vec3(0.9, 0.9, 0.9), specular=0.3, roughness=0.5,
        reflectivity=0.2, checker_scale=2.0, checker_color=Vec3(0.4, 0.4, 0.4),
    )
    scene.add_object(lambda p: sdf_plane(p, height=0.0), ground_mat)

    # Gold sphere
    gold_mat = Material(
        albedo=Vec3(1.0, 0.76, 0.33), specular=0.9, roughness=0.15,
        reflectivity=0.8, fresnel=0.5,
    )
    scene.add_object(lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0), gold_mat)

    # Glass sphere
    glass_mat = Material(
        albedo=Vec3(0.95, 0.95, 0.98), specular=0.9, roughness=0.02,
        reflectivity=0.3, fresnel=0.04, transparency=0.9, ior=1.5,
    )
    scene.add_object(lambda p: sdf_sphere(p, Vec3(3, 1, 1), 1.0), glass_mat)

    # Matte red box
    red_mat = Material(albedo=Vec3(0.8, 0.1, 0.1), specular=0.1, roughness=0.9)
    scene.add_object(lambda p: sdf_box(p, Vec3(-3, 0.6, 1), Vec3(0.6, 0.6, 0.6)), red_mat)

    # Emissive sphere
    emissive_mat = Material(emissive=Vec3(0.8, 0.9, 1.0))
    scene.add_object(lambda p: sdf_sphere(p, Vec3(0, 0.5, -3), 0.5), emissive_mat)

    # Three-point lighting
    scene.add_directional_light(Vec3(0.5, 0.8, 0.3), Vec3(1.0, 0.95, 0.85), 1.0)  # Key light
    scene.add_directional_light(Vec3(-0.7, 0.3, 0.5), Vec3(0.4, 0.5, 0.7), 0.4)  # Fill light
    scene.add_point_light(Vec3(0, 5, -5), Vec3(1.0, 0.95, 0.9), 3.0)  # Back light

    return scene


# Scene registry
SCENES = {
    'demo': build_demo_scene,
    'chess': build_chess_scene,
    'terrain': build_terrain_scene,
    'abstract': build_abstract_scene,
    'glass': build_glass_scene,
    'fractal': build_fractal_scene,
    'studio': build_studio_scene,
}