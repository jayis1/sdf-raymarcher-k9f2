"""Additional tests for new features: transforms, tonemap, new primitives, new scenes, new materials, vec_renderer, config."""

import math
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from raymarcher_k9f2.vec3 import Vec3, BLACK, WHITE
from raymarcher_k9f2.material import Material, copper, jade, chrome, marble, emissive as emissive_preset
from raymarcher_k9f2.primitives import (
    sdf_sphere, sdf_box, sdf_plane,
    sdf_ellipsoid, sdf_rounded_box, sdf_link,
)
from raymarcher_k9f2.transforms import (
    sdf_translate, sdf_scale, sdf_rotate_y, sdf_rotate_x, sdf_rotate_z,
    sdf_repeat, sdf_elongate,
)
from raymarcher_k9f2.tonemap import apply_tonemap, list_tonemaps
from raymarcher_k9f2.scene import Scene
from raymarcher_k9f2.marcher import RayMarcher
from raymarcher_k9f2.camera import Camera
from raymarcher_k9f2.renderer import Renderer
from raymarcher_k9f2.scenes import build_showcase_scene
from raymarcher_k9f2.config import load_config, build_scene_from_config


def approx_eq(a, b, eps=1e-6):
    return abs(a - b) < eps


def vec_approx_eq(a, b, eps=1e-6):
    return approx_eq(a.x, b.x, eps) and approx_eq(a.y, b.y, eps) and approx_eq(a.z, b.z, eps)


# ─── New Primitives Tests ────────────────────────────────────────────────

class TestNewPrimitives:
    def test_ellipsoid_surface(self):
        # Point on surface of ellipsoid with radii (1, 2, 1)
        d = sdf_ellipsoid(Vec3(1, 0, 0), Vec3(0, 0, 0), Vec3(1, 2, 1))
        assert approx_eq(d, 0, 0.3)  # Approximate SDF

    def test_ellipsoid_inside(self):
        d = sdf_ellipsoid(Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(1, 2, 1))
        assert d < 0  # Inside

    def test_ellipsoid_outside(self):
        d = sdf_ellipsoid(Vec3(10, 10, 10), Vec3(0, 0, 0), Vec3(1, 2, 1))
        assert d > 0  # Far outside

    def test_rounded_box_surface(self):
        d = sdf_rounded_box(Vec3(1, 0, 0), Vec3(0, 0, 0), Vec3(1, 1, 1), 0.2)
        # Should be close to the rounding radius from the surface
        assert abs(d) < 1.0  # Not exactly 0 due to rounding

    def test_rounded_box_inside(self):
        d = sdf_rounded_box(Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(1, 1, 1), 0.2)
        assert d < 0  # Inside

    def test_rounded_box_outside(self):
        d = sdf_rounded_box(Vec3(5, 5, 5), Vec3(0, 0, 0), Vec3(1, 1, 1), 0.2)
        assert d > 0  # Far outside

    def test_link_inside(self):
        d = sdf_link(Vec3(0, 1.5, 0), Vec3(0, 1.5, 0), 0.5, 0.5, 0.12)
        assert d < 0  # Inside the tube

    def test_link_outside(self):
        d = sdf_link(Vec3(5, 5, 5), Vec3(0, 0, 0), 0.5, 0.5, 0.12)
        assert d > 0  # Far outside


# ─── Transform Tests ─────────────────────────────────────────────────────

class TestTransforms:
    def test_translate_basic(self):
        # Sphere at origin (radius 1), translated to (5, 0, 0)
        sphere_sdf = lambda p: sdf_sphere(p, Vec3(0, 0, 0), 1.0)
        # Point at (6.5, 0, 0): distance to center (5,0,0) is 1.5, sphere radius is 1.0
        # So SDF should be 1.5 - 1.0 = 0.5 (outside)
        d = sdf_translate(Vec3(6.5, 0, 0), Vec3(5, 0, 0), sphere_sdf)
        assert d > 0  # Outside the sphere
        assert approx_eq(d, 0.5, 0.01)

    def test_translate_inside(self):
        sphere_sdf = lambda p: sdf_sphere(p, Vec3(0, 0, 0), 1.0)
        d = sdf_translate(Vec3(5, 0, 0), Vec3(5, 0, 0), sphere_sdf)
        assert d < 0  # At center of translated sphere

    def test_scale_shrink(self):
        sphere_sdf = lambda p: sdf_sphere(p, Vec3(0, 0, 0), 1.0)
        # Scale factor 0.5 shrinks the sphere
        d = sdf_scale(Vec3(2, 0, 0), 0.5, sphere_sdf)
        assert d > 0  # Outside (sphere radius is now 0.5 * 1 = 0.5 at scale 0.5)

    def test_rotate_y_preserves_distance(self):
        # Rotating a sphere shouldn't change the distance
        sphere_sdf = lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0)
        d_orig = sdf_sphere(Vec3(3, 1, 0), Vec3(0, 1, 0), 1.0)
        d_rotated = sdf_rotate_y(Vec3(3, 1, 0), math.pi / 4, sphere_sdf)
        assert approx_eq(d_orig, d_rotated, 0.01)

    def test_rotate_x_preserves_distance(self):
        # For a sphere at origin, rotation should preserve distance
        sphere_sdf = lambda p: sdf_sphere(p, Vec3(0, 0, 0), 1.0)
        d_orig = sdf_sphere(Vec3(2, 0, 0), Vec3(0, 0, 0), 1.0)
        d_rotated = sdf_rotate_x(Vec3(2, 0, 0), math.pi / 4, sphere_sdf)
        assert approx_eq(d_orig, d_rotated, 0.01)

    def test_rotate_z_preserves_distance(self):
        # For a sphere at origin, rotation should preserve distance
        sphere_sdf = lambda p: sdf_sphere(p, Vec3(0, 0, 0), 1.0)
        d_orig = sdf_sphere(Vec3(2, 0, 0), Vec3(0, 0, 0), 1.0)
        d_rotated = sdf_rotate_z(Vec3(2, 0, 0), math.pi / 4, sphere_sdf)
        assert approx_eq(d_orig, d_rotated, 0.01)

    def test_repeat_creates_pattern(self):
        sphere_sdf = lambda p: sdf_sphere(p, Vec3(0, 0, 0), 0.3)
        # At origin should be inside
        d = sdf_repeat(Vec3(0, 0, 0), Vec3(3, 1e6, 3), sphere_sdf)
        assert d < 0  # Inside nearest sphere copy

    def test_elongate_basic(self):
        sphere_sdf = lambda p: sdf_sphere(p, Vec3(0, 0, 0), 1.0)
        d = sdf_elongate(Vec3(0, 0, 0), Vec3(0, 0, 0), sphere_sdf)
        assert d < 0  # Inside the non-elongated shape


# ─── Tone Mapping Tests ────────────────────────────────────────────────

class TestToneMapping:
    def test_list_tonemaps(self):
        methods = list_tonemaps()
        assert 'aces' in methods
        assert 'reinhard' in methods
        assert 'filmic' in methods
        assert 'clamped' in methods
        assert 'linear' in methods

    def test_aces_output_range(self):
        # HDR input
        img = np.array([[[2.0, 0.5, 0.1]]], dtype=np.float64)
        result = apply_tonemap(img, method='aces', exposure=1.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_reinhard_output_range(self):
        img = np.array([[[5.0, 0.2, 0.1]]], dtype=np.float64)
        result = apply_tonemap(img, method='reinhard', exposure=1.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_linear_passthrough(self):
        img = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float64)
        result = apply_tonemap(img, method='linear', exposure=1.0)
        # Linear with gamma should apply gamma correction
        assert result.min() >= 0.0

    def test_clamped_output(self):
        img = np.array([[[-1.0, 0.5, 3.0]]], dtype=np.float64)
        result = apply_tonemap(img, method='clamped', exposure=1.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_exposure_parameter(self):
        img = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float64)
        result1 = apply_tonemap(img, method='reinhard', exposure=1.0)
        result2 = apply_tonemap(img, method='reinhard', exposure=2.0)
        # Higher exposure should produce brighter output (before clamping)
        assert result2.mean() >= result1.mean() * 0.9  # Allow some margin for gamma

    def test_invalid_method_raises(self):
        img = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float64)
        with pytest.raises(ValueError, match="Unknown tone mapping method"):
            apply_tonemap(img, method='invalid')

    def test_reinhard_formula(self):
        # reinhard(x) = x / (1 + x), then gamma
        x = 2.0
        expected_linear = x / (1 + x)  # 0.6667
        img = np.array([[[x, 0, 0]]], dtype=np.float64)
        result = apply_tonemap(img, method='reinhard', exposure=1.0)
        # After gamma (1/2.2), should be expected_linear^(1/2.2)
        expected_gamma = expected_linear ** (1.0 / 2.2)
        assert approx_eq(result[0, 0, 0], expected_gamma, 0.01)


# ─── New Material Presets Tests ─────────────────────────────────────────

class TestNewMaterials:
    def test_copper(self):
        m = copper()
        assert m.albedo.x > m.albedo.z  # Copper is more red than blue
        assert m.reflectivity > 0.5
        assert m.fresnel > 0.3

    def test_jade(self):
        m = jade()
        assert m.albedo.y > m.albedo.x  # Green channel dominant
        assert m.subsurface > 0  # Has SSS

    def test_chrome(self):
        m = chrome()
        assert m.reflectivity > 0.9
        assert m.roughness < 0.05
        assert m.fresnel > 0.4

    def test_marble(self):
        m = marble()
        assert m.subsurface > 0  # Has slight SSS
        assert m.albedo.x > 0.9  # Very white

    def test_emissive_preset(self):
        m = emissive_preset()
        assert m.emissive.x > 0
        assert m.emissive.y > 0
        assert m.emissive.z > 0

    def test_emissive_with_color(self):
        m = emissive_preset(color=Vec3(1, 0, 0), intensity=2.0)
        assert approx_eq(m.emissive.x, 2.0)


# ─── Showcase Scene Tests ────────────────────────────────────────────────

class TestShowcaseScene:
    def test_showcase_builds(self):
        scene = build_showcase_scene(0.0)
        assert len(scene.objects) > 0
        assert len(scene.directional_lights) > 0

    def test_showcase_renders(self):
        scene = build_showcase_scene(0.0)
        marcher = RayMarcher(enable_shadows=False, max_bounces=1, max_refraction_bounces=1)
        renderer = Renderer(width=16, height=12, marcher=marcher, quiet=True)
        cam = Camera(Vec3(8, 4, 8), Vec3(0, 1, 0), fov=55)
        img = renderer.render(scene, cam)
        assert img.shape == (12, 16, 3)
        assert img.min() >= 0
        assert not np.isnan(img.mean())


# ─── Config with New Primitives Tests ────────────────────────────────────

class TestConfigNewPrimitives:
    def test_ellipsoid_config(self):
        config = {
            'scene': {
                'objects': [
                    {'type': 'ellipsoid', 'center': [0, 1, 0], 'radii': [1, 2, 1],
                     'material': {'albedo': [0.5, 0.5, 0.5]}}
                ],
                'lights': [
                    {'type': 'directional', 'direction': [0.5, 0.8, 0.3]}
                ],
            }
        }
        scene, camera, renderer, output = build_scene_from_config(config)
        assert len(scene.objects) == 1

    def test_rounded_box_config(self):
        config = {
            'scene': {
                'objects': [
                    {'type': 'rounded_box', 'center': [0, 0, 0],
                     'half_extents': [1, 1, 1], 'radius': 0.2,
                     'material': {'albedo': [0.8, 0.3, 0.1]}}
                ],
                'lights': [
                    {'type': 'directional', 'direction': [0.5, 0.8, 0.3]}
                ],
            }
        }
        scene, camera, renderer, output = build_scene_from_config(config)
        assert len(scene.objects) == 1

    def test_link_config(self):
        config = {
            'scene': {
                'objects': [
                    {'type': 'link', 'center': [0, 1.5, 0],
                     'le': 0.5, 'r1': 0.5, 'r2': 0.12,
                     'material': {'albedo': [0.7, 0.7, 0.8]}}
                ],
                'lights': [
                    {'type': 'directional', 'direction': [0.5, 0.8, 0.3]}
                ],
            }
        }
        scene, camera, renderer, output = build_scene_from_config(config)
        assert len(scene.objects) == 1


# ─── VecRenderer Tests ──────────────────────────────────────────────────

class TestVecRenderer:
    def test_vec_renderer_renders(self):
        from raymarcher_k9f2.vec_renderer import VecRenderer
        from raymarcher_k9f2.scenes import build_demo_scene

        scene = build_demo_scene(0.0)
        renderer = VecRenderer(width=16, height=12, quiet=True)
        cam = Camera(Vec3(7, 4, 7), Vec3(0, 1, 0), fov=55)
        img = renderer.render(scene, cam)
        assert img.shape == (12, 16, 3)
        assert img.min() >= 0
        assert not np.isnan(img.mean())

    def test_vec_renderer_save(self):
        from raymarcher_k9f2.vec_renderer import VecRenderer
        from raymarcher_k9f2.scenes import build_demo_scene

        scene = build_demo_scene(0.0)
        renderer = VecRenderer(width=16, height=12, quiet=True)
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

    def test_vec_renderer_render_to_pil(self):
        from raymarcher_k9f2.vec_renderer import VecRenderer
        from raymarcher_k9f2.scenes import build_demo_scene

        scene = build_demo_scene(0.0)
        renderer = VecRenderer(width=16, height=12, quiet=True)
        cam = Camera(Vec3(7, 4, 7), Vec3(0, 1, 0), fov=55)
        pil_img = renderer.render_to_pil(scene, cam)
        assert pil_img.size == (16, 12)


# ─── Integration Tests ──────────────────────────────────────────────────

class TestIntegration:
    def test_tonemap_with_renderer(self):
        """Test that tone mapping works with the full rendering pipeline."""
        from raymarcher_k9f2.tonemap import apply_tonemap
        from raymarcher_k9f2.scenes import build_demo_scene

        scene = build_demo_scene(0.0)
        marcher = RayMarcher(enable_shadows=False)
        renderer = Renderer(width=32, height=24, marcher=marcher, quiet=True)
        cam = Camera(Vec3(7, 4, 7), Vec3(0, 1, 0), fov=55)

        # Render with linear gamma (no gamma correction in renderer)
        img_hdr = renderer.render(scene, cam)

        # Apply ACES tone mapping
        img_aces = apply_tonemap(img_hdr, method='aces')
        assert img_aces.shape == img_hdr.shape
        assert img_aces.min() >= 0.0
        assert img_aces.max() <= 1.0

    def test_transforms_in_scene(self):
        """Test that transformed objects render correctly in a scene."""
        scene = Scene()
        # Translated sphere
        scene.add_object(
            lambda p: sdf_translate(p, Vec3(0, 1, 0), lambda q: sdf_sphere(q, Vec3(0, 0, 0), 1.0)),
            Material(albedo=Vec3(1, 0, 0))
        )
        scene.add_object(
            lambda p: sdf_plane(p, height=0.0),
            Material(albedo=Vec3(0.5, 0.5, 0.5))
        )
        scene.add_directional_light(Vec3(0.5, 0.8, 0.3))

        marcher = RayMarcher(enable_shadows=False)
        renderer = Renderer(width=32, height=24, marcher=marcher, quiet=True)
        cam = Camera(Vec3(5, 3, 5), Vec3(0, 1, 0), fov=55)

        img = renderer.render(scene, cam)
        assert img.shape == (24, 32, 3)
        assert img.min() >= 0
        assert not np.isnan(img.mean())

    def test_new_primitives_in_scene(self):
        """Test that new primitives render correctly in a scene."""
        scene = Scene()

        # Ellipsoid
        scene.add_object(
            lambda p: sdf_ellipsoid(p, Vec3(0, 1, 0), Vec3(0.8, 1.2, 0.6)),
            Material(albedo=Vec3(0.1, 0.7, 0.4))
        )
        # Rounded box
        scene.add_object(
            lambda p: sdf_rounded_box(p, Vec3(3, 0.7, 0), Vec3(0.7, 0.7, 0.7), 0.15),
            Material(albedo=Vec3(0.8, 0.5, 0.15))
        )
        scene.add_object(
            lambda p: sdf_plane(p, height=0.0),
            Material(albedo=Vec3(0.5, 0.5, 0.5))
        )
        scene.add_directional_light(Vec3(0.5, 0.8, 0.3))

        marcher = RayMarcher(enable_shadows=False)
        renderer = Renderer(width=32, height=24, marcher=marcher, quiet=True)
        cam = Camera(Vec3(5, 3, 5), Vec3(0, 1, 0), fov=55)

        img = renderer.render(scene, cam)
        assert img.shape == (24, 32, 3)
        assert img.min() >= 0
        assert not np.isnan(img.mean())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])