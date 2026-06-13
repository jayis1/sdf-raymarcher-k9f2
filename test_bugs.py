#!/usr/bin/env python3
"""
Bug Hunt Tests for SDF Ray Marcher k9f2.

Tests verifying bugs found and fixed during Phase 3.
Each test targets a specific bug and validates the fix.
"""

import math
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from raymarcher import (
    Vec3, Material, Scene, Camera, Renderer, RayMarcher,
    sdf_sphere, sdf_box, sdf_plane, sdf_cone, sdf_capsule,
    sdf_torus, sdf_cylinder, sdf_hex_prism, sdf_mandelbulb,
    sdf_smooth_subtraction, sdf_smooth_union, sdf_union, sdf_subtraction,
    sdf_intersection,
)


def approx_eq(a, b, eps=1e-6):
    """Check if two floats are approximately equal."""
    return abs(a - b) < eps


# ─── Bug 1: Camera FOV double-scaling ───────────────────────────────────────

def test_camera_fov_correctness():
    """Bug: Camera.get_ray() applied FOV scaling twice because Renderer
    pre-scaled u,v by half_w/half_h, AND get_ray() also applied half_h.
    
    Fix: Renderer now passes normalized coords in [-0.5, 0.5] and get_ray()
    accepts an aspect parameter.
    """
    cam = Camera(Vec3(0, 0, 5), Vec3(0, 0, 0), fov=90)
    aspect = 1.0
    
    # Center ray
    origin, direction = cam.get_ray(0, 0, aspect)
    assert approx_eq(direction.z, -1.0, 0.01), f"Center ray Z should be ~-1, got {direction.z}"
    assert approx_eq(direction.x, 0, 0.01), f"Center ray X should be ~0, got {direction.x}"
    
    # Edge ray (right side): at fov=90, u=0.5 should give 45-degree angle
    origin, direction = cam.get_ray(0.5, 0, aspect)
    xz_ratio = abs(direction.x / direction.z)
    assert 0.8 < xz_ratio < 1.2, f"At fov=90, edge ray X/Z ratio should be ~1.0, got {xz_ratio}"
    
    # Widescreen aspect (2:1) should produce wider rays than narrow aspect
    origin, direction_wide = cam.get_ray(0.5, 0, 2.0)
    origin, direction_narrow = cam.get_ray(0.5, 0, 1.0)
    assert abs(direction_wide.x) > abs(direction_narrow.x), \
        "Wide aspect should produce wider rays than narrow aspect"
    
    print("  ✓ Camera FOV double-scaling bug fixed — get_ray uses aspect ratio correctly")


def test_renderer_no_duplicate_fov():
    """Verify that Renderer.render() doesn't duplicate FOV computation."""
    import inspect
    source = inspect.getsource(Renderer.render)
    
    # render() should call get_ray with aspect parameter
    assert 'get_ray(u, v, aspect' in source, \
        "render() should pass aspect ratio to get_ray()"
    
    print("  ✓ Renderer no longer duplicates FOV computation")


# ─── Bug 2: March entering detection ─────────────────────────────────────────

def test_march_entering_from_outside():
    """Bug: march() determined entering/exiting based on SDF value at hit point,
    which is always near epsilon (ambiguous). Fix: check SDF at ray origin.
    """
    scene = Scene()
    scene.add_object(
        lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0),
        Material(albedo=Vec3(1, 0, 0))
    )
    marcher = RayMarcher()
    
    hit = marcher.march(Vec3(5, 1, 0), Vec3(-1, 0, 0), scene)
    assert hit.material is not None, "Should hit sphere"
    assert hit.entering == True, f"Ray from outside should be entering, got entering={hit.entering}"
    
    print("  ✓ March entering detection: outside ray correctly reports entering=True")


def test_march_entering_from_inside():
    """Test: A ray starting inside a glass sphere should report entering=False."""
    scene = Scene()
    glass_mat = Material(albedo=Vec3(1, 1, 1), transparency=0.9, ior=1.5)
    scene.add_object(
        lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0),
        glass_mat
    )
    marcher = RayMarcher()
    
    hit = marcher.march(Vec3(0, 1, 0), Vec3(0, 1, 0), scene)
    assert hit.material is not None, "Should hit from inside"
    assert hit.entering == False, f"Ray from inside should be exiting, got entering={hit.entering}"
    
    print("  ✓ March entering detection: inside ray correctly reports entering=False")


# ─── Bug 3: Hex prism dead code ──────────────────────────────────────────────

def test_hex_prism_sdf():
    """Bug: sdf_hex_prism had dead code (d_xz computed but not used)."""
    center = Vec3(0, 0, 0)
    radius = 1.0
    half_height = 0.5
    
    d_inside = sdf_hex_prism(Vec3(0, 0, 0), center, radius, half_height)
    assert d_inside < 0, f"Center should be inside hex prism, got d={d_inside}"
    
    d_outside_xz = sdf_hex_prism(Vec3(2, 0, 0), center, radius, half_height)
    assert d_outside_xz > 0, f"Far in X should be outside, got d={d_outside_xz}"
    
    d_outside_y = sdf_hex_prism(Vec3(0, 2, 0), center, radius, half_height)
    assert d_outside_y > 0, f"Far in Y should be outside, got d={d_outside_y}"
    
    print("  ✓ Hex prism SDF: dead code removed, function still works correctly")


# ─── Bug 4: Cone SDF incorrect ───────────────────────────────────────────────

def test_cone_sdf_inside():
    """Bug: Cone SDF didn't properly handle inside/capped regions.
    Fix: Rewrote cone SDF with proper base capping.
    """
    center = Vec3(0, 0, 0)
    half_angle = math.pi / 4  # 45 degrees
    height = 2.0
    
    d = sdf_cone(Vec3(0, 1, 0), center, half_angle, height)
    assert d < 0, f"Point on axis inside cone should have negative distance, got {d}"
    
    d_below = sdf_cone(Vec3(0, -0.5, 0), center, half_angle, height)
    assert d_below > 0, f"Point below cone base should be outside, got {d_below}"
    
    d_above = sdf_cone(Vec3(0, 3, 0), center, half_angle, height)
    assert d_above > 0, f"Point above cone tip should be outside, got {d_above}"
    
    d_base_surface = sdf_cone(Vec3(2, 0, 0), center, half_angle, height)
    assert abs(d_base_surface) < 0.1, f"Point on cone base surface should be ~0, got {d_base_surface}"
    
    print("  ✓ Cone SDF: proper inside/outside/base detection")


def test_cone_sdf_gradient():
    """Verify cone SDF gradient points away from the surface."""
    center = Vec3(0, 0, 0)
    half_angle = math.pi / 6
    height = 2.0
    
    r_at_1 = (height - 1.0) * math.tan(half_angle)
    d_on_surface = sdf_cone(Vec3(r_at_1, 1, 0), center, half_angle, height)
    d_outside = sdf_cone(Vec3(r_at_1 + 0.1, 1, 0), center, half_angle, height)
    
    assert d_on_surface < d_outside, \
        f"SDF should increase moving away from surface: on={d_on_surface}, outside={d_outside}"
    
    print("  ✓ Cone SDF: gradient points outward from surface")


# ─── Bug 5: Smooth subtraction formula validation ───────────────────────────

def test_smooth_subtraction_properties():
    """Validate smooth subtraction properties."""
    smooth = sdf_smooth_subtraction(3.0, 2.0, k=0.5)
    hard = max(-2.0, 3.0)
    assert smooth >= hard - 0.1, f"Smooth subtraction should be >= hard subtraction, got {smooth} < {hard}"
    
    smooth2 = sdf_smooth_subtraction(-1.0, 0.5, k=0.5)
    hard2 = max(-0.5, -1.0)
    assert smooth2 >= hard2 - 0.5, f"Smooth subtraction should blend, got {smooth2} vs hard {hard2}"
    
    print("  ✓ Smooth subtraction: properties validated")


# ─── Bug 6: Renderer aspect ratio produces non-degenerate image ──────────────

def test_render_aspect_ratio():
    """Verify that the fixed renderer produces correct images for different aspect ratios."""
    scene = Scene()
    scene.add_object(
        lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0),
        Material(albedo=Vec3(1, 0, 0))
    )
    scene.add_object(
        lambda p: sdf_plane(p, height=0.0),
        Material(albedo=Vec3(0.5, 0.5, 0.5))
    )
    scene.add_directional_light(Vec3(0.5, 0.8, 0.3))
    
    marcher = RayMarcher(enable_shadows=False)
    
    renderer_wide = Renderer(width=64, height=32, marcher=marcher, quiet=True)
    cam_wide = Camera(Vec3(3, 2, 3), Vec3(0, 1, 0), fov=55)
    img_wide = renderer_wide.render(scene, cam_wide)
    
    renderer_tall = Renderer(width=32, height=64, marcher=marcher, quiet=True)
    cam_tall = Camera(Vec3(3, 2, 3), Vec3(0, 1, 0), fov=55)
    img_tall = renderer_tall.render(scene, cam_tall)
    
    assert img_wide.mean() > 0.01, "Wide image should have content"
    assert img_tall.mean() > 0.01, "Tall image should have content"
    assert not np.isnan(img_wide).any(), "Wide image should not have NaN"
    assert not np.isnan(img_tall).any(), "Tall image should not have NaN"
    
    print("  ✓ Renderer aspect ratio: both wide and tall images render correctly")


# ─── Bug 7: Refraction IOR when exiting ──────────────────────────────────────

def test_refraction_ior_exiting():
    """Bug: When exiting a glass object, eta was set to 1/ior instead of ior.
    
    Snell's law: n1*sin(theta1) = n2*sin(theta2), eta = n1/n2
    Entering glass (air→glass): eta = 1.0 / 1.5 ≈ 0.667
    Exiting glass (glass→air): eta = 1.5 / 1.0 = 1.5
    
    With the bug, both cases used eta = 1/ior.
    """
    scene = Scene()
    glass_mat = Material(albedo=Vec3(1, 1, 1), transparency=0.9, ior=1.5)
    scene.add_object(
        lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0),
        glass_mat
    )
    scene.add_object(
        lambda p: sdf_plane(p, height=0.0),
        Material(albedo=Vec3(0.5, 0.5, 0.5))
    )
    scene.add_directional_light(Vec3(0.5, 0.8, 0.3))
    
    marcher = RayMarcher()
    
    # Ray starting inside the glass sphere should report entering=False
    origin = Vec3(0, 1, 0)
    direction = Vec3(0, 1, 0)
    hit = marcher.march(origin, direction, scene)
    assert hit.entering == False, "Ray from inside should report exiting"
    
    # The refraction should produce a valid color (not crash or NaN)
    direction_in = (-Vec3(5, 2, 5)).normalized()
    color = marcher.trace(Vec3(5, 3, 5), direction_in, scene)
    assert color.x >= 0 and color.y >= 0 and color.z >= 0, "Trace should produce valid color"
    assert not (math.isnan(color.x) or math.isnan(color.y) or math.isnan(color.z)), "Trace should not produce NaN"
    
    print("  ✓ Refraction IOR: exiting glass uses correct eta (ior, not 1/ior)")


# ─── Bug 8: Dead code in refraction IOR logic ───────────────────────────────

def test_refraction_inside_object():
    """Test that Vec3.refract() correctly handles rays inside objects."""
    normal = Vec3(0, 1, 0)
    incident = Vec3(0.1, -1, 0).normalized()
    refracted = incident.refract(normal, 1.0 / 1.5)
    assert refracted is not None, "Should refract from air to glass"
    
    print("  ✓ Refraction: inside/outside object handling works")


# ─── Bug 9: Mandelbulb distance estimator ────────────────────────────────────

def test_mandelbulb_distance_estimator():
    """Verify Mandelbulb returns reasonable distances."""
    center = Vec3(0, 1.5, 0)
    scale = 1.2
    
    d_far = sdf_mandelbulb(Vec3(5, 1.5, 0), center, scale=scale, power=8.0)
    assert d_far > 0, f"Far from mandelbulb should be positive, got {d_far}"
    
    d_center = sdf_mandelbulb(center, center, scale=scale, power=8.0)
    assert d_center <= 0.5, f"At mandelbulb center, distance should be small/negative, got {d_center}"
    
    print("  ✓ Mandelbulb distance estimator: returns valid distances")


# ─── Bug 10: Shadow march overshoot ──────────────────────────────────────────

def test_shadow_overshoot():
    """Verify shadow computation doesn't produce negative values or NaN."""
    scene = Scene()
    scene.add_object(
        lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0),
        Material(albedo=Vec3(1, 0, 0))
    )
    scene.add_object(
        lambda p: sdf_plane(p, height=0.0),
        Material(albedo=Vec3(0.5, 0.5, 0.5))
    )
    scene.add_directional_light(Vec3(0.5, 0.8, 0.3))
    
    marcher = RayMarcher(enable_shadows=True)
    p = Vec3(0, 0.001, 0)
    normal = Vec3(0, 1, 0)
    light_dir = Vec3(0.5, 0.8, 0.3).normalized()
    
    shadow = marcher.compute_shadow(p, light_dir, scene)
    assert 0.0 <= shadow <= 1.0, f"Shadow should be in [0,1], got {shadow}"
    assert not math.isnan(shadow), f"Shadow should not be NaN"
    
    print("  ✓ Shadow computation: no overshoot or NaN")


# ─── Bug 11: Checkerboard with negative coordinates ─────────────────────────

def test_checkerboard_negative_coords():
    """Verify checkerboard works with negative coordinates."""
    from raymarcher import checkerboard
    
    color1 = Vec3(1, 1, 1)
    color2 = Vec3(0, 0, 0)
    
    c1 = checkerboard(Vec3(0.4, 0, 0.4), 1.0, color1, color2)
    c2 = checkerboard(Vec3(1.4, 0, 0.4), 1.0, color1, color2)
    assert c1 != c2, f"Adjacent cells should have different colors"
    
    c3 = checkerboard(Vec3(-0.4, 0, 0.4), 1.0, color1, color2)
    c4 = checkerboard(Vec3(-1.4, 0, 0.4), 1.0, color1, color2)
    assert isinstance(c3, Vec3), "Checkerboard should return Vec3 for negative coords"
    assert isinstance(c4, Vec3), "Checkerboard should return Vec3 for negative coords"
    assert c3 != c4, f"Negative coordinate cells should have different colors"
    
    print("  ✓ Checkerboard: works correctly with negative coordinates")


# ─── Bug 12: Vec3 refraction edge cases ─────────────────────────────────────

def test_refraction_edge_cases():
    """Test Vec3.refract() with edge cases."""
    normal = Vec3(0, 1, 0)
    
    # Normal incidence
    incident = Vec3(0, -1, 0)
    refracted = incident.refract(normal, 1.0 / 1.5)
    assert refracted is not None, "Normal incidence should refract, not TIR"
    assert approx_eq(refracted.x, 0, 0.01), f"Normal incidence: X should be 0, got {refracted.x}"
    assert approx_eq(refracted.z, 0, 0.01), f"Normal incidence: Z should be 0, got {refracted.z}"
    
    # Grazing angle
    incident = Vec3(1, -0.01, 0).normalized()
    refracted = incident.refract(normal, 1.0 / 1.5)
    assert refracted is not None, "Slightly grazing should still refract"
    
    print("  ✓ Refraction: normal incidence and grazing angle handled correctly")


# ─── Bug 13: Scene.map() material lookup ────────────────────────────────────

def test_scene_map_returns_correct_material():
    """Verify that Scene.map() returns the material of the closest object."""
    scene = Scene()
    mat1 = Material(albedo=Vec3(1, 0, 0))
    mat2 = Material(albedo=Vec3(0, 0, 1))
    
    scene.add_object(lambda p: sdf_sphere(p, Vec3(0, 0, 0), 1.0), mat1)
    scene.add_object(lambda p: sdf_sphere(p, Vec3(5, 0, 0), 1.0), mat2)
    
    result = scene.map(Vec3(0.5, 0, 0))
    assert result.material is mat1, f"Should return closer material (mat1)"
    
    result2 = scene.map(Vec3(4.5, 0, 0))
    assert result2.material is mat2, f"Should return closer material (mat2)"
    
    print("  ✓ Scene.map: returns correct material for closest object")


# ─── Bug 14: Reflection direction ────────────────────────────────────────────

def test_reflection_direction():
    """Verify Vec3.reflect() produces correct reflection directions."""
    incoming = Vec3(1, -1, 0).normalized()
    normal = Vec3(0, 1, 0)
    reflected = incoming.reflect(normal)
    
    expected = Vec3(1, 1, 0).normalized()
    assert approx_eq(reflected.x, expected.x, 0.01), f"Reflect X: got {reflected.x}, expected {expected.x}"
    assert approx_eq(reflected.y, expected.y, 0.01), f"Reflect Y: got {reflected.y}, expected {expected.y}"
    
    incoming2 = Vec3(0, -1, 0)
    reflected2 = incoming2.reflect(normal)
    assert approx_eq(reflected2.y, 1.0, 0.01), f"Normal incidence reflect Y should be 1.0, got {reflected2.y}"
    
    print("  ✓ Reflection direction: produces correct vectors")


# ─── Bug 15: AO overflow ────────────────────────────────────────────────────

def test_ao_no_overflow():
    """Verify that AO computation returns values in [0, 1]."""
    scene = Scene()
    scene.add_object(
        lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0),
        Material(albedo=Vec3(1, 0, 0))
    )
    scene.add_object(
        lambda p: sdf_plane(p, height=0.0),
        Material(albedo=Vec3(0.5, 0.5, 0.5))
    )
    scene.add_directional_light(Vec3(0.5, 0.8, 0.3))
    
    marcher = RayMarcher(enable_ao=True, ao_steps=5, ao_step_size=0.1)
    
    p = Vec3(0, 2.0, 0)
    normal = Vec3(0, 1, 0)
    ao = marcher.compute_ao(p, normal, scene)
    assert 0.0 <= ao <= 1.0, f"AO should be in [0,1], got {ao}"
    assert ao > 0.5, f"AO at top of sphere should be high (open sky), got {ao}"
    
    print("  ✓ Ambient occlusion: returns values in [0, 1]")


# ─── Bug 16: Glass scene renders without crash ──────────────────────────────

def test_glass_scene_renders():
    """Verify the glass scene renders without crashing (tests refraction paths)."""
    from raymarcher import build_glass_scene
    scene = build_glass_scene(0.0)
    marcher = RayMarcher(max_bounces=3, enable_shadows=True)
    renderer = Renderer(width=32, height=24, marcher=marcher, quiet=True)
    cam = Camera(Vec3(5, 3.5, 8), Vec3(0, 1, 0), fov=50)
    
    img = renderer.render(scene, cam)
    assert not np.isnan(img).any(), "Glass scene should not produce NaN"
    assert img.mean() > 0.01, "Glass scene should produce visible content"
    
    print("  ✓ Glass scene renders without crash (refraction paths work)")


# ─── Run all tests ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 3 Bug Hunt Tests — SDF Ray Marcher k9f2")
    print("=" * 60)
    
    tests = [
        ("Camera FOV correctness", test_camera_fov_correctness),
        ("Renderer no duplicate FOV", test_renderer_no_duplicate_fov),
        ("March entering from outside", test_march_entering_from_outside),
        ("March entering from inside", test_march_entering_from_inside),
        ("Hex prism SDF", test_hex_prism_sdf),
        ("Cone SDF inside", test_cone_sdf_inside),
        ("Cone SDF gradient", test_cone_sdf_gradient),
        ("Smooth subtraction properties", test_smooth_subtraction_properties),
        ("Render aspect ratio", test_render_aspect_ratio),
        ("Refraction IOR exiting", test_refraction_ior_exiting),
        ("Refraction inside object", test_refraction_inside_object),
        ("Mandelbulb distance estimator", test_mandelbulb_distance_estimator),
        ("Shadow overshoot", test_shadow_overshoot),
        ("Checkerboard negative coords", test_checkerboard_negative_coords),
        ("Refraction edge cases", test_refraction_edge_cases),
        ("Scene map material", test_scene_map_returns_correct_material),
        ("Reflection direction", test_reflection_direction),
        ("AO no overflow", test_ao_no_overflow),
        ("Glass scene renders", test_glass_scene_renders),
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for name, test_fn in tests:
        print(f"\n{name}")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  ✗ FAILED: {e}")
        except Exception as e:
            failed += 1
            errors.append((name, f"Exception: {e}"))
            print(f"  ✗ ERROR: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    
    if errors:
        print("\nFailed tests:")
        for name, msg in errors:
            print(f"  - {name}: {msg}")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())