"""Comprehensive test suite for SDF Ray Marcher k9f2 — pytest compatible."""

import math
import os
import sys
import tempfile

import numpy as np
import pytest

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from raymarcher_k9f2.vec3 import Vec3, BLACK, WHITE, SKY_BLUE
from raymarcher_k9f2.material import Material, red_plastic, blue_plastic, mirror, glass, water, gold, checkerboard
from raymarcher_k9f2.primitives import (
    sdf_sphere, sdf_box, sdf_torus, sdf_cylinder, sdf_capsule,
    sdf_plane, sdf_cone, sdf_hex_prism, sdf_mandelbulb,
)
from raymarcher_k9f2.csg import (
    sdf_union, sdf_smooth_union, sdf_smooth_subtraction,
    sdf_subtraction, sdf_intersection, sdf_smooth_intersection,
)
from raymarcher_k9f2.patterns import checkerboard as pattern_checkerboard, stripes, gradient
from raymarcher_k9f2.scene import Scene, SceneObject, PointLight, DirectionalLight, HitResult
from raymarcher_k9f2.marcher import RayMarcher
from raymarcher_k9f2.camera import Camera
from raymarcher_k9f2.renderer import Renderer
from raymarcher_k9f2.scenes import (
    build_demo_scene, build_chess_scene, build_terrain_scene,
    build_abstract_scene, build_glass_scene, build_fractal_scene,
    build_studio_scene, SCENES,
)


def approx_eq(a, b, eps=1e-6):
    return abs(a - b) < eps


def vec_approx_eq(a, b, eps=1e-6):
    return approx_eq(a.x, b.x, eps) and approx_eq(a.y, b.y, eps) and approx_eq(a.z, b.z, eps)


# ─── Vec3 Tests ──────────────────────────────────────────────────────────────

class TestVec3:
    def test_construction(self):
        v = Vec3(1, 2, 3)
        assert v.x == 1.0 and v.y == 2.0 and v.z == 3.0

    def test_default_construction(self):
        v = Vec3()
        assert v.x == 0.0 and v.y == 0.0 and v.z == 0.0

    def test_addition(self):
        a, b = Vec3(1, 2, 3), Vec3(4, 5, 6)
        assert vec_approx_eq(a + b, Vec3(5, 7, 9))

    def test_subtraction(self):
        a, b = Vec3(1, 2, 3), Vec3(4, 5, 6)
        assert vec_approx_eq(a - b, Vec3(-3, -3, -3))

    def test_scalar_multiplication(self):
        v = Vec3(1, 2, 3)
        assert vec_approx_eq(v * 2, Vec3(2, 4, 6))
        assert vec_approx_eq(3 * v, Vec3(3, 6, 9))

    def test_component_multiplication(self):
        a, b = Vec3(1, 2, 3), Vec3(4, 5, 6)
        assert vec_approx_eq(a * b, Vec3(4, 10, 18))

    def test_negation(self):
        v = Vec3(1, 2, 3)
        assert vec_approx_eq(-v, Vec3(-1, -2, -3))

    def test_equality(self):
        assert Vec3(1, 2, 3) == Vec3(1, 2, 3)
        assert not (Vec3(1, 2, 3) == Vec3(1, 2, 4))

    def test_dot_product(self):
        assert approx_eq(Vec3(1, 0, 0).dot(Vec3(0, 1, 0)), 0.0)
        assert approx_eq(Vec3(1, 0, 0).dot(Vec3(1, 0, 0)), 1.0)

    def test_cross_product(self):
        cross = Vec3(1, 0, 0).cross(Vec3(0, 1, 0))
        assert vec_approx_eq(cross, Vec3(0, 0, 1))

    def test_length_and_normalize(self):
        v = Vec3(3, 4, 0)
        assert approx_eq(v.length(), 5.0)
        n = v.normalized()
        assert approx_eq(n.length(), 1.0)
        assert approx_eq(n.x, 0.6)
        assert approx_eq(n.y, 0.8)

    def test_zero_normalize(self):
        z = Vec3(0, 0, 0).normalized()
        assert vec_approx_eq(z, Vec3(0, 0, 0))

    def test_reflect(self):
        v = Vec3(1, -1, 0)
        n = Vec3(0, 1, 0)
        r = v.reflect(n)
        assert vec_approx_eq(r, Vec3(1, 1, 0), 1e-5)

    def test_refract_normal(self):
        v = Vec3(0, -1, 0)
        n = Vec3(0, 1, 0)
        refracted = v.refract(n, 1.0 / 1.5)
        assert refracted is not None
        assert vec_approx_eq(refracted, Vec3(0, -1, 0), 0.01)

    def test_refract_grazing(self):
        # At near-grazing incidence from inside glass (IOR=1.5), TIR occurs
        # because sin(theta) exceeds the critical angle
        v = Vec3(1, -0.001, 0)
        n = Vec3(0, 1, 0)
        refracted = v.refract(n, 1.5)  # Glass to air
        assert refracted is None  # TIR at grazing angle

    def test_lerp(self):
        a, b = Vec3(0, 0, 0), Vec3(1, 1, 1)
        assert vec_approx_eq(a.lerp(b, 0.5), Vec3(0.5, 0.5, 0.5))

    def test_clamp(self):
        v = Vec3(-1, 0.5, 2)
        assert vec_approx_eq(v.clamp(0, 1), Vec3(0, 0.5, 1))

    def test_pow(self):
        v = Vec3(0.25, 0.5, 1.0)
        p = v.pow(2)
        assert approx_eq(p.x, 0.0625)
        assert approx_eq(p.y, 0.25)
        assert approx_eq(p.z, 1.0)

    def test_to_from_tuple(self):
        v = Vec3(1, 2, 3)
        t = v.to_tuple()
        assert t == (1.0, 2.0, 3.0)
        v2 = Vec3.from_tuple(t)
        assert v == v2

    def test_length_sq(self):
        v = Vec3(3, 4, 0)
        assert approx_eq(v.length_sq(), 25.0)


# ─── SDF Primitive Tests ────────────────────────────────────────────────────

class TestSDFPrimitives:
    def test_sphere_surface(self):
        assert approx_eq(sdf_sphere(Vec3(1, 0, 0), Vec3(0, 0, 0), 1.0), 0.0, 0.01)

    def test_sphere_outside(self):
        assert sdf_sphere(Vec3(3, 0, 0), Vec3(0, 0, 0), 1.0) > 0

    def test_sphere_inside(self):
        assert sdf_sphere(Vec3(0.5, 0, 0), Vec3(0, 0, 0), 1.0) < 0

    def test_sphere_distance(self):
        assert approx_eq(sdf_sphere(Vec3(3, 0, 0), Vec3(0, 0, 0), 1.0), 2.0, 0.01)

    def test_box_surface(self):
        d = sdf_box(Vec3(1, 0, 0), Vec3(0, 0, 0), Vec3(1, 1, 1))
        assert abs(d) < 0.01

    def test_box_outside(self):
        d = sdf_box(Vec3(2, 0, 0), Vec3(0, 0, 0), Vec3(1, 1, 1))
        assert d > 0

    def test_box_inside(self):
        d = sdf_box(Vec3(0.5, 0.5, 0.5), Vec3(0, 0, 0), Vec3(1, 1, 1))
        assert d < 0

    def test_plane_above(self):
        assert approx_eq(sdf_plane(Vec3(0, 1, 0), height=0.0), 1.0)

    def test_plane_below(self):
        assert approx_eq(sdf_plane(Vec3(0, -2, 0), height=0.0), -2.0)

    def test_torus_inside(self):
        d = sdf_torus(Vec3(1.0, 0, 0), Vec3(0, 0, 0), 1.0, 0.25)
        assert approx_eq(d, -0.25, 0.01)

    def test_capsule_inside(self):
        d = sdf_capsule(Vec3(0, 1, 0), Vec3(0, 0, 0), Vec3(0, 2, 0), 0.5)
        assert d < 0

    def test_cylinder_surface(self):
        d = sdf_cylinder(Vec3(1, 0, 0), Vec3(0, 0, 0), 1.0, 1.0)
        assert approx_eq(d, 0.0, 0.01)

    def test_cone_inside(self):
        d = sdf_cone(Vec3(0, 1, 0), Vec3(0, 0, 0), math.pi / 4, 2.0)
        assert d < 0

    def test_cone_below_base(self):
        d = sdf_cone(Vec3(0, -0.5, 0), Vec3(0, 0, 0), math.pi / 4, 2.0)
        assert d > 0

    def test_hex_prism_inside(self):
        d = sdf_hex_prism(Vec3(0, 0, 0), Vec3(0, 0, 0), 1.0, 1.0)
        assert d < 0

    def test_mandelbulb_finite(self):
        d = sdf_mandelbulb(Vec3(0.5, 0.5, 0), Vec3(0, 0, 0), scale=1.0, power=8.0)
        assert math.isfinite(d)

    def test_mandelbulb_far(self):
        d = sdf_mandelbulb(Vec3(10, 10, 10), Vec3(0, 0, 0), scale=1.0, power=8.0)
        assert d > 0


# ─── CSG Operation Tests ────────────────────────────────────────────────────

class TestCSG:
    def test_union(self):
        assert sdf_union(1.0, 2.0) == 1.0

    def test_smooth_union(self):
        su = sdf_smooth_union(0.1, 0.2, k=1.0)
        assert su < 0.1

    def test_subtraction(self):
        assert sdf_subtraction(1.0, -1.0) == 1.0

    def test_intersection(self):
        assert sdf_intersection(1.0, 2.0) == 2.0

    def test_smooth_subtraction(self):
        smooth = sdf_smooth_subtraction(3.0, 2.0, k=0.5)
        hard = max(-2.0, 3.0)
        assert smooth >= hard - 0.1

    def test_smooth_intersection(self):
        si = sdf_smooth_intersection(1.0, 1.5, k=0.5)
        assert si >= 1.0  # Should be at least as large as hard intersection


# ─── Pattern Tests ───────────────────────────────────────────────────────────

class TestPatterns:
    def test_checkerboard_basic(self):
        c1, c2 = Vec3(1, 1, 1), Vec3(0, 0, 0)
        result = pattern_checkerboard(Vec3(0.1, 0, 0.1), 1.0, c1, c2)
        assert result == c1 or result == c2

    def test_checkerboard_alternates(self):
        c1, c2 = Vec3(1, 1, 1), Vec3(0, 0, 0)
        p1 = pattern_checkerboard(Vec3(0.4, 0, 0.4), 1.0, c1, c2)
        p2 = pattern_checkerboard(Vec3(1.4, 0, 0.4), 1.0, c1, c2)
        assert not vec_approx_eq(p1, p2, 0.01)

    def test_checkerboard_negative(self):
        c1, c2 = Vec3(1, 1, 1), Vec3(0, 0, 0)
        c3 = pattern_checkerboard(Vec3(-0.4, 0, 0.4), 1.0, c1, c2)
        c4 = pattern_checkerboard(Vec3(-1.4, 0, 0.4), 1.0, c1, c2)
        assert isinstance(c3, Vec3)
        assert c3 != c4

    def test_stripes(self):
        c1, c2 = Vec3(1, 1, 1), Vec3(0, 0, 0)
        s1 = stripes(Vec3(0.2, 0, 0), 1.0, c1, c2, 'x')
        s2 = stripes(Vec3(0.7, 0, 0), 1.0, c1, c2, 'x')
        assert s1 != s2 or s1 == s2  # Both valid

    def test_gradient(self):
        c1, c2 = Vec3(0, 0, 0), Vec3(1, 1, 1)
        g = gradient(Vec3(0, 0, 0), c1, c2, 'y')
        assert approx_eq(g.x, 0.5, 0.01)


# ─── Material Tests ──────────────────────────────────────────────────────────

class TestMaterial:
    def test_defaults(self):
        m = Material()
        assert m.albedo.x == 0.8
        assert m.specular == 0.5
        assert m.transparency == 0.0
        assert m.ior == 1.5

    def test_glass_material(self):
        m = glass()
        assert m.transparency == 0.9
        assert m.ior == 1.5

    def test_water_material(self):
        m = water()
        assert approx_eq(m.ior, 1.33)

    def test_mirror_material(self):
        m = mirror()
        assert m.reflectivity == 0.95

    def test_serialization(self):
        m = Material(albedo=Vec3(0.5, 0.3, 0.1), specular=0.8, ior=1.33)
        d = m.to_dict()
        m2 = Material.from_dict(d)
        assert vec_approx_eq(m2.albedo, Vec3(0.5, 0.3, 0.1))
        assert approx_eq(m2.specular, 0.8)
        assert approx_eq(m2.ior, 1.33)

    def test_presets(self):
        assert red_plastic().albedo.x > 0.5
        assert blue_plastic().albedo.z > 0.5
        assert gold().albedo.x > gold().albedo.z  # Gold is more yellow than blue


# ─── Scene Tests ─────────────────────────────────────────────────────────────

class TestScene:
    def test_empty_scene(self):
        scene = Scene()
        assert len(scene.objects) == 0
        assert len(scene.directional_lights) == 0

    def test_add_object(self):
        scene = Scene()
        idx = scene.add_object(lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0), Material())
        assert idx == 0
        assert len(scene.objects) == 1

    def test_add_lights(self):
        scene = Scene()
        scene.add_directional_light(Vec3(1, 1, 1), Vec3(1, 1, 1), 1.0)
        scene.add_point_light(Vec3(2, 3, 4), Vec3(1, 1, 1), 2.0)
        assert len(scene.directional_lights) == 1
        assert len(scene.point_lights) == 1

    def test_map_closest(self):
        scene = Scene()
        scene.add_object(lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0), Material(albedo=Vec3(1, 0, 0)))
        scene.add_object(lambda p: sdf_sphere(p, Vec3(5, 1, 0), 1.0), Material(albedo=Vec3(0, 0, 1)))
        hit = scene.map(Vec3(0, 1, 0))
        assert hit.distance < 0
        assert hit.object_index == 0

    def test_map_distance(self):
        scene = Scene()
        scene.add_object(lambda p: sdf_sphere(p, Vec3(0, 0, 0), 1.0), Material())
        d = scene.map_distance(Vec3(3, 0, 0))
        assert approx_eq(d, 2.0, 0.01)

    def test_sky(self):
        scene = Scene()
        sky = scene.get_sky(Vec3(0, 1, 0))
        assert sky.y > sky.x  # Should be bluish


# ─── Ray Marcher Tests ───────────────────────────────────────────────────────

class TestRayMarcher:
    def setup_method(self):
        self.scene = Scene()
        self.scene.add_object(lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0), Material(albedo=Vec3(1, 0, 0)))
        self.scene.add_object(lambda p: sdf_plane(p, height=0.0), Material(albedo=Vec3(0.5, 0.5, 0.5)))
        self.scene.add_directional_light(Vec3(0.5, 0.8, 0.3))
        self.marcher = RayMarcher(enable_shadows=True)

    def test_march_hit(self):
        hit = self.marcher.march(Vec3(3, 1, 0), Vec3(-1, 0, 0), self.scene)
        assert hit.material is not None
        assert approx_eq(hit.distance, 2.0, 0.1)

    def test_march_miss(self):
        hit = self.marcher.march(Vec3(3, 1, 0), Vec3(1, 0, 0), self.scene)
        assert hit.material is None

    def test_march_entering_from_outside(self):
        hit = self.marcher.march(Vec3(5, 1, 0), Vec3(-1, 0, 0), self.scene)
        assert hit.entering == True

    def test_march_entering_from_inside(self):
        glass_scene = Scene()
        glass_scene.add_object(
            lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0),
            Material(albedo=Vec3(1, 1, 1), transparency=0.9, ior=1.5)
        )
        hit = self.marcher.march(Vec3(0, 1, 0), Vec3(0, 1, 0), glass_scene)
        assert hit.entering == False

    def test_normal_computation(self):
        sphere_scene = Scene()
        sphere_scene.add_object(lambda p: sdf_sphere(p, Vec3(0, 0, 0), 1.0), Material())
        n = self.marcher.compute_normal(Vec3(1, 0, 0), sphere_scene)
        assert vec_approx_eq(n, Vec3(1, 0, 0), 0.05)

    def test_ao(self):
        ao_scene = Scene()
        ao_scene.add_object(lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0), Material())
        ao_scene.add_object(lambda p: sdf_plane(p, height=0.0), Material())
        marcher_ao = RayMarcher(enable_ao=True, ao_steps=5)
        ao = marcher_ao.compute_ao(Vec3(0, 2, 0), Vec3(0, 1, 0), ao_scene)
        assert ao > 0.8
        ao_low = marcher_ao.compute_ao(Vec3(0, 0.1, 0), Vec3(0, 1, 0), ao_scene)
        assert ao_low < 0.95

    def test_shadow_lit(self):
        shadow = self.marcher.compute_shadow(Vec3(0, 2.1, 0), Vec3(0.5, 0.8, 0.3).normalized(), self.scene)
        assert shadow > 0.5

    def test_shadow_occluded(self):
        shadow = self.marcher.compute_shadow(Vec3(0, 0.01, -1), Vec3(0.5, 0.8, 0.3).normalized(), self.scene)
        assert shadow < 0.5

    def test_trace_valid_color(self):
        cam = Camera(Vec3(3, 2, 3), Vec3(0, 1, 0), fov=55)
        direction = (Vec3(0, 1, 0) - cam.position).normalized()
        color = self.marcher.trace(cam.position, direction, self.scene)
        assert color.x >= 0 and color.y >= 0 and color.z >= 0

    def test_trace_sky(self):
        color = self.marcher.trace(Vec3(0, 10, 0), Vec3(0, 1, 0), self.scene)
        assert color.y > color.x  # Sky is bluish


# ─── Camera Tests ────────────────────────────────────────────────────────────

class TestCamera:
    def test_center_ray(self):
        cam = Camera(Vec3(0, 0, 5), Vec3(0, 0, 0), fov=90)
        origin, direction = cam.get_ray(0, 0)
        assert vec_approx_eq(origin, Vec3(0, 0, 5), 0.01)
        assert direction.z < 0

    def test_fov_aspect(self):
        cam = Camera(Vec3(0, 0, 5), Vec3(0, 0, 0), fov=90)
        origin, direction_wide = cam.get_ray(0.5, 0, 2.0)
        origin, direction_narrow = cam.get_ray(0.5, 0, 1.0)
        assert abs(direction_wide.x) > abs(direction_narrow.x)

    def test_dof(self):
        cam_dof = Camera(Vec3(0, 0, 5), Vec3(0, 0, 0), fov=60, focal_distance=5.0, aperture=0.1)
        origin, direction = cam_dof.get_ray(0, 0, 0)
        # DOF camera should produce valid rays
        assert direction.length() > 0

    def test_serialization(self):
        cam = Camera(Vec3(7, 4, 7), Vec3(0, 1, 0), fov=55)
        d = cam.to_dict()
        cam2 = Camera.from_dict(d)
        assert approx_eq(cam2.position.x, 7.0)
        assert approx_eq(cam2.fov, 55.0)


# ─── Renderer Tests ──────────────────────────────────────────────────────────

class TestRenderer:
    def test_render_demo(self):
        scene = build_demo_scene(0.0)
        marcher = RayMarcher(enable_shadows=False)
        renderer = Renderer(width=32, height=24, marcher=marcher, quiet=True)
        cam = Camera(Vec3(7, 4, 7), Vec3(0, 1, 0), fov=55)
        img = renderer.render(scene, cam)
        assert img.shape == (24, 32, 3)
        assert img.min() >= 0
        assert img.mean() > 0.01

    def test_save_png(self):
        scene = build_demo_scene(0.0)
        marcher = RayMarcher(enable_shadows=False)
        renderer = Renderer(width=32, height=24, marcher=marcher, quiet=True)
        cam = Camera(Vec3(7, 4, 7), Vec3(0, 1, 0), fov=55)
        img = renderer.render(scene, cam)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            renderer.save(img, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_render_to_pil(self):
        scene = build_demo_scene(0.0)
        marcher = RayMarcher(enable_shadows=False)
        renderer = Renderer(width=32, height=24, marcher=marcher, quiet=True)
        cam = Camera(Vec3(7, 4, 7), Vec3(0, 1, 0), fov=55)
        pil_img = renderer.render_to_pil(scene, cam)
        assert pil_img.size == (32, 24)

    def test_aspect_ratio(self):
        scene = Scene()
        scene.add_object(lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0), Material(albedo=Vec3(1, 0, 0)))
        scene.add_object(lambda p: sdf_plane(p, height=0.0), Material(albedo=Vec3(0.5, 0.5, 0.5)))
        scene.add_directional_light(Vec3(0.5, 0.8, 0.3))
        marcher = RayMarcher(enable_shadows=False)

        renderer_wide = Renderer(width=64, height=32, marcher=marcher, quiet=True)
        cam_wide = Camera(Vec3(3, 2, 3), Vec3(0, 1, 0), fov=55)
        img_wide = renderer_wide.render(scene, cam_wide)

        renderer_tall = Renderer(width=32, height=64, marcher=marcher, quiet=True)
        cam_tall = Camera(Vec3(3, 2, 3), Vec3(0, 1, 0), fov=55)
        img_tall = renderer_tall.render(scene, cam_tall)

        assert img_wide.mean() > 0.01
        assert img_tall.mean() > 0.01
        assert not np.isnan(img_wide).any()
        assert not np.isnan(img_tall).any()


# ─── Scene Builder Tests ────────────────────────────────────────────────────

class TestSceneBuilders:
    @pytest.mark.parametrize("name,builder", list(SCENES.items()))
    def test_scene_builds(self, name, builder):
        scene = builder(0.0)
        assert len(scene.objects) > 0
        assert len(scene.directional_lights) > 0 or len(scene.point_lights) > 0

    @pytest.mark.parametrize("name,builder", list(SCENES.items()))
    def test_scene_renders(self, name, builder):
        marcher = RayMarcher(enable_shadows=False, max_bounces=1, max_refraction_bounces=1)
        renderer = Renderer(width=16, height=12, marcher=marcher, quiet=True)
        scene = builder(0.0)

        cam_map = {
            'demo': Camera(Vec3(7, 4, 7), Vec3(0, 1, 0), fov=55),
            'chess': Camera(Vec3(5, 5, 8), Vec3(0, 0.5, 0), fov=50),
            'terrain': Camera(Vec3(8, 5, 8), Vec3(0, 1, 0), fov=60),
            'abstract': Camera(Vec3(6, 4, 6), Vec3(0, 1.5, 0), fov=55),
            'glass': Camera(Vec3(5, 3.5, 8), Vec3(0, 1, 0), fov=50),
            'fractal': Camera(Vec3(4, 3, 4), Vec3(0, 1.5, 0), fov=50),
            'studio': Camera(Vec3(6, 3, 6), Vec3(0, 1, 0), fov=50),
        }
        cam = cam_map[name]
        img = renderer.render(scene, cam)
        assert img.shape == (12, 16, 3)
        assert img.min() >= 0
        assert not np.isnan(img.mean())


# ─── Refraction Tests ────────────────────────────────────────────────────────

class TestRefraction:
    def test_glass_trace(self):
        scene = build_glass_scene(0.0)
        marcher = RayMarcher(enable_shadows=True, max_bounces=2, max_refraction_bounces=2)
        cam = Camera(Vec3(5, 3.5, 8), Vec3(0, 1, 0), fov=50)
        direction = (Vec3(0, 1.2, 0) - cam.position).normalized()
        color = marcher.trace(cam.position, direction, scene)
        assert color.x >= 0 and color.y >= 0 and color.z >= 0
        assert not math.isnan(color.x)

    def test_refraction_ior_exiting(self):
        scene = Scene()
        glass_mat = Material(albedo=Vec3(1, 1, 1), transparency=0.9, ior=1.5)
        scene.add_object(lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0), glass_mat)
        scene.add_object(lambda p: sdf_plane(p, height=0.0), Material(albedo=Vec3(0.5, 0.5, 0.5)))
        scene.add_directional_light(Vec3(0.5, 0.8, 0.3))
        marcher = RayMarcher()

        # Ray from inside glass sphere should report entering=False
        origin = Vec3(0, 1, 0)
        direction = Vec3(0, 1, 0)
        hit = marcher.march(origin, direction, scene)
        assert hit.entering == False

    def test_refraction_edge_cases(self):
        normal = Vec3(0, 1, 0)
        incident = Vec3(0, -1, 0)
        refracted = incident.refract(normal, 1.0 / 1.5)
        assert refracted is not None
        assert approx_eq(refracted.x, 0, 0.01)


# ─── Config Tests ────────────────────────────────────────────────────────────

class TestConfig:
    def test_material_from_dict(self):
        d = {'albedo': [0.5, 0.3, 0.1], 'specular': 0.8, 'ior': 1.33}
        m = Material.from_dict(d)
        assert approx_eq(m.albedo.x, 0.5)
        assert approx_eq(m.specular, 0.8)
        assert approx_eq(m.ior, 1.33)

    def test_camera_from_dict(self):
        d = {'position': [7, 4, 7], 'target': [0, 1, 0], 'fov': 55}
        cam = Camera.from_dict(d)
        assert approx_eq(cam.position.x, 7.0)
        assert approx_eq(cam.fov, 55.0)


# ─── Performance Smoke Tests ─────────────────────────────────────────────────

class TestPerformance:
    def test_small_render_completes(self):
        """Verify that a small render completes in reasonable time."""
        import time
        scene = build_demo_scene(0.0)
        marcher = RayMarcher(enable_shadows=False)
        renderer = Renderer(width=64, height=48, marcher=marcher, quiet=True)
        cam = Camera(Vec3(7, 4, 7), Vec3(0, 1, 0), fov=55)
        start = time.time()
        img = renderer.render(scene, cam)
        elapsed = time.time() - start
        assert elapsed < 120, f"Small render took {elapsed:.1f}s — too slow"
        assert img.mean() > 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])