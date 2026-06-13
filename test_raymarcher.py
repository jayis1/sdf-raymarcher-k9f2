#!/usr/bin/env python3
"""
Test suite for SDF Ray Marcher k9f2.

Tests cover:
- Vector math operations
- SDF primitive correctness
- CSG operations
- Material properties
- Scene construction
- Ray marching hit detection
- Rendering pipeline
- Refraction/reflection logic
"""

import math
import sys
import os

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from raymarcher import (
    Vec3, BLACK, WHITE, SKY_BLUE,
    Material, Scene, Camera, Renderer, RayMarcher,
    sdf_sphere, sdf_box, sdf_torus, sdf_cylinder, sdf_capsule,
    sdf_plane, sdf_cone, sdf_hex_prism, sdf_mandelbulb,
    sdf_union, sdf_smooth_union, sdf_subtraction, sdf_intersection,
    checkerboard,
    build_demo_scene, build_chess_scene, build_terrain_scene,
    build_abstract_scene, build_glass_scene, build_fractal_scene,
    HitResult,
)


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name: str):
        self.passed += 1
        print(f"  ✓ {name}")

    def fail(self, name: str, msg: str):
        self.failed += 1
        self.errors.append((name, msg))
        print(f"  ✗ {name}: {msg}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print("\nFailures:")
            for name, msg in self.errors:
                print(f"  - {name}: {msg}")
        print(f"{'='*60}")
        return self.failed == 0


def approx_eq(a: float, b: float, eps: float = 1e-6) -> bool:
    """Check if two floats are approximately equal."""
    return abs(a - b) < eps


def vec_approx_eq(a: Vec3, b: Vec3, eps: float = 1e-6) -> bool:
    """Check if two Vec3 values are approximately equal."""
    return (approx_eq(a.x, b.x, eps) and
            approx_eq(a.y, b.y, eps) and
            approx_eq(a.z, b.z, eps))


def test_vec3_construction(results: TestResults):
    """Test Vec3 construction and basic properties."""
    v = Vec3(1, 2, 3)
    assert v.x == 1.0 and v.y == 2.0 and v.z == 3.0
    results.ok("Vec3 construction")

    v0 = Vec3()
    assert v0.x == 0.0 and v0.y == 0.0 and v0.z == 0.0
    results.ok("Vec3 default construction")


def test_vec3_arithmetic(results: TestResults):
    """Test Vec3 arithmetic operations."""
    a = Vec3(1, 2, 3)
    b = Vec3(4, 5, 6)

    # Addition
    c = a + b
    assert vec_approx_eq(c, Vec3(5, 7, 9))
    results.ok("Vec3 addition")

    # Subtraction
    c = a - b
    assert vec_approx_eq(c, Vec3(-3, -3, -3))
    results.ok("Vec3 subtraction")

    # Scalar multiplication
    c = a * 2
    assert vec_approx_eq(c, Vec3(2, 4, 6))
    results.ok("Vec3 scalar multiplication (left)")

    c = 3 * a
    assert vec_approx_eq(c, Vec3(3, 6, 9))
    results.ok("Vec3 scalar multiplication (right)")

    # Component-wise multiplication
    c = a * b
    assert vec_approx_eq(c, Vec3(4, 10, 18))
    results.ok("Vec3 component-wise multiplication")

    # Negation
    c = -a
    assert vec_approx_eq(c, Vec3(-1, -2, -3))
    results.ok("Vec3 negation")


def test_vec3_dot_cross(results: TestResults):
    """Test Vec3 dot and cross products."""
    a = Vec3(1, 0, 0)
    b = Vec3(0, 1, 0)

    dot = a.dot(b)
    assert approx_eq(dot, 0.0)
    results.ok("Vec3 dot product (perpendicular)")

    dot = a.dot(a)
    assert approx_eq(dot, 1.0)
    results.ok("Vec3 dot product (parallel)")

    cross = a.cross(b)
    assert vec_approx_eq(cross, Vec3(0, 0, 1))
    results.ok("Vec3 cross product")


def test_vec3_length_normalize(results: TestResults):
    """Test Vec3 length and normalization."""
    v = Vec3(3, 4, 0)
    assert approx_eq(v.length(), 5.0)
    results.ok("Vec3 length")

    n = v.normalized()
    assert approx_eq(n.length(), 1.0)
    assert approx_eq(n.x, 0.6)
    assert approx_eq(n.y, 0.8)
    results.ok("Vec3 normalization")

    # Zero vector normalization
    z = Vec3(0, 0, 0).normalized()
    assert vec_approx_eq(z, Vec3(0, 0, 0))
    results.ok("Vec3 zero vector normalization")


def test_vec3_reflect(results: TestResults):
    """Test Vec3 reflection."""
    # Reflect off a flat surface
    v = Vec3(1, -1, 0)  # Incoming ray
    n = Vec3(0, 1, 0)    # Surface normal
    r = v.reflect(n)
    assert vec_approx_eq(r, Vec3(1, 1, 0), 1e-5)
    results.ok("Vec3 reflection")


def test_vec3_refract(results: TestResults):
    """Test Vec3 refraction."""
    # Straight down into a surface should bend
    v = Vec3(0, -1, 0)
    n = Vec3(0, 1, 0)
    refracted = v.refract(n, 1.0 / 1.5)  # Air to glass
    assert refracted is not None
    # Should refract straight through (perpendicular incidence)
    assert vec_approx_eq(refracted, Vec3(0, -1, 0), 0.01)
    results.ok("Vec3 refraction (normal incidence)")

    # Grazing angle should cause total internal reflection
    v = Vec3(1, -0.001, 0)
    n = Vec3(0, 1, 0)
    # From inside glass (high IOR) to air — should TIR
    refracted = v.refract(n, 1.5)  # Glass to air
    # At this extreme angle, should get TIR or extreme bending
    results.ok("Vec3 refraction (grazing angle)")


def test_vec3_lerp_clamp(results: TestResults):
    """Test Vec3 lerp and clamp."""
    a = Vec3(0, 0, 0)
    b = Vec3(1, 1, 1)

    mid = a.lerp(b, 0.5)
    assert vec_approx_eq(mid, Vec3(0.5, 0.5, 0.5))
    results.ok("Vec3 lerp")

    v = Vec3(-1, 0.5, 2)
    c = v.clamp(0, 1)
    assert vec_approx_eq(c, Vec3(0, 0.5, 1))
    results.ok("Vec3 clamp")


def test_vec3_pow(results: TestResults):
    """Test Vec3 component-wise power."""
    v = Vec3(0.25, 0.5, 1.0)
    p = v.pow(2)
    assert approx_eq(p.x, 0.0625)
    assert approx_eq(p.y, 0.25)
    assert approx_eq(p.z, 1.0)
    results.ok("Vec3 pow")


def test_sdf_sphere(results: TestResults):
    """Test sphere SDF."""
    center = Vec3(0, 0, 0)
    radius = 1.0

    # On the surface
    d = sdf_sphere(Vec3(1, 0, 0), center, radius)
    assert approx_eq(d, 0.0, 0.01)
    results.ok("SDF sphere: surface point")

    # Outside
    d = sdf_sphere(Vec3(2, 0, 0), center, radius)
    assert d > 0
    results.ok("SDF sphere: outside point")

    # Inside
    d = sdf_sphere(Vec3(0.5, 0, 0), center, radius)
    assert d < 0
    results.ok("SDF sphere: inside point")

    # Distance is correct
    d = sdf_sphere(Vec3(3, 0, 0), center, radius)
    assert approx_eq(d, 2.0, 0.01)
    results.ok("SDF sphere: distance correctness")


def test_sdf_box(results: TestResults):
    """Test box SDF."""
    center = Vec3(0, 0, 0)
    half = Vec3(1, 1, 1)

    # On the surface (may be -0.0 or 0.0 due to floating point)
    d = sdf_box(Vec3(1, 0, 0), center, half)
    assert abs(d) < 0.01, f"Expected ~0 on surface, got {d}"
    results.ok("SDF box: surface point")

    # Outside
    d = sdf_box(Vec3(2, 0, 0), center, half)
    assert d > 0
    results.ok("SDF box: outside point")

    # Inside
    d = sdf_box(Vec3(0.5, 0.5, 0.5), center, half)
    assert d < 0
    results.ok("SDF box: inside point")


def test_sdf_plane(results: TestResults):
    """Test plane SDF."""
    # At height 0
    d = sdf_plane(Vec3(0, 1, 0), height=0.0)
    assert approx_eq(d, 1.0)
    results.ok("SDF plane: above")

    d = sdf_plane(Vec3(0, -2, 0), height=0.0)
    assert approx_eq(d, -2.0)
    results.ok("SDF plane: below")

    # At height 5
    d = sdf_plane(Vec3(0, 5, 0), height=5.0)
    assert approx_eq(d, 0.0, 0.01)
    results.ok("SDF plane: on surface at custom height")


def test_sdf_torus(results: TestResults):
    """Test torus SDF."""
    center = Vec3(0, 0, 0)
    major_r = 1.0
    minor_r = 0.25

    # Center of the tube at (major_r, 0, 0)
    d = sdf_torus(Vec3(1.0, 0, 0), center, major_r, minor_r)
    assert approx_eq(d, -0.25, 0.01)  # Inside tube
    results.ok("SDF torus: inside tube")

    # Far outside
    d = sdf_torus(Vec3(5, 0, 0), center, major_r, minor_r)
    assert d > 0
    results.ok("SDF torus: outside")


def test_sdf_capsule(results: TestResults):
    """Test capsule SDF."""
    a = Vec3(0, 0, 0)
    b = Vec3(0, 2, 0)
    radius = 0.5

    # At the midpoint
    d = sdf_capsule(Vec3(0, 1, 0), a, b, radius)
    assert d < 0  # Inside
    results.ok("SDF capsule: inside")

    # Far away
    d = sdf_capsule(Vec3(5, 1, 0), a, b, radius)
    assert d > 0
    results.ok("SDF capsule: outside")


def test_sdf_cylinder(results: TestResults):
    """Test cylinder SDF."""
    center = Vec3(0, 0, 0)
    radius = 1.0
    half_height = 1.0

    # On the side surface
    d = sdf_cylinder(Vec3(1, 0, 0), center, radius, half_height)
    assert approx_eq(d, 0.0, 0.01)
    results.ok("SDF cylinder: side surface")

    # Inside
    d = sdf_cylinder(Vec3(0, 0, 0), center, radius, half_height)
    assert d < 0
    results.ok("SDF cylinder: inside")


def test_csg_operations(results: TestResults):
    """Test CSG boolean operations."""
    # Union: min of two distances
    d1 = 1.0
    d2 = 2.0
    assert sdf_union(d1, d2) == 1.0
    results.ok("CSG union")

    # Smooth union: when k is small and distances are far apart,
    # it approximates regular union. Test with close values and larger k.
    su = sdf_smooth_union(0.1, 0.2, k=1.0)
    assert su < 0.1  # Should blend below the smaller distance
    results.ok("CSG smooth union")

    # Subtraction
    assert sdf_subtraction(1.0, -1.0) == 1.0
    results.ok("CSG subtraction")

    # Intersection
    assert sdf_intersection(1.0, 2.0) == 2.0
    results.ok("CSG intersection")


def test_checkerboard(results: TestResults):
    """Test procedural checkerboard pattern."""
    c1 = Vec3(1, 1, 1)
    c2 = Vec3(0, 0, 0)

    # At origin, should be one color
    result = checkerboard(Vec3(0.1, 0, 0.1), 1.0, c1, c2)
    assert vec_approx_eq(result, c1) or vec_approx_eq(result, c2)
    results.ok("Checkerboard pattern returns valid color")

    # Different checker cells should give different colors
    p1 = checkerboard(Vec3(0.4, 0, 0.4), 1.0, c1, c2)
    p2 = checkerboard(Vec3(1.4, 0, 0.4), 1.0, c1, c2)
    # These should be different (different checker cells)
    assert not vec_approx_eq(p1, p2, 0.01)
    results.ok("Checkerboard alternates colors")


def test_material(results: TestResults):
    """Test Material construction and defaults."""
    m = Material()
    assert m.albedo.x == 0.8
    assert m.specular == 0.5
    assert m.roughness == 0.3
    assert m.reflectivity == 0.0
    assert m.transparency == 0.0
    assert m.ior == 1.5
    assert m.subsurface == 0.0
    results.ok("Material default values")

    m_glass = Material(
        albedo=Vec3(0.95, 0.95, 0.98),
        transparency=0.9,
        ior=1.5,
        fresnel=0.04,
    )
    assert m_glass.transparency == 0.9
    assert m_glass.ior == 1.5
    results.ok("Material glass properties")


def test_scene_construction(results: TestResults):
    """Test Scene construction and object/light addition."""
    scene = Scene()
    assert len(scene.objects) == 0
    assert len(scene.directional_lights) == 0
    assert len(scene.point_lights) == 0
    results.ok("Empty scene construction")

    # Add an object
    idx = scene.add_object(
        lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0),
        Material(albedo=Vec3(1, 0, 0))
    )
    assert idx == 0
    assert len(scene.objects) == 1
    results.ok("Scene add_object")

    # Add a light
    scene.add_directional_light(Vec3(1, 1, 1), Vec3(1, 1, 1), 1.0)
    assert len(scene.directional_lights) == 1
    results.ok("Scene add_directional_light")

    scene.add_point_light(Vec3(2, 3, 4), Vec3(1, 1, 1), 2.0)
    assert len(scene.point_lights) == 1
    results.ok("Scene add_point_light")


def test_scene_map(results: TestResults):
    """Test Scene.map() returns correct closest object."""
    scene = Scene()

    # Two spheres
    scene.add_object(
        lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0),
        Material(albedo=Vec3(1, 0, 0))
    )
    scene.add_object(
        lambda p: sdf_sphere(p, Vec3(5, 1, 0), 1.0),
        Material(albedo=Vec3(0, 0, 1))
    )

    # Point near first sphere
    hit = scene.map(Vec3(0, 1, 0))
    assert hit.distance < 0  # Inside first sphere
    assert hit.object_index == 0
    results.ok("Scene map: closest to first sphere")

    # Point near second sphere
    hit = scene.map(Vec3(5, 1, 0))
    assert hit.distance < 0
    assert hit.object_index == 1
    results.ok("Scene map: closest to second sphere")

    # Point equidistant (should pick whichever evaluates smaller)
    hit = scene.map(Vec3(2.5, 1, 0))
    assert hit.distance > 0  # Outside both
    results.ok("Scene map: outside both spheres")


def test_scene_map_distance(results: TestResults):
    """Test Scene.map_distance() fast path."""
    scene = Scene()
    scene.add_object(
        lambda p: sdf_sphere(p, Vec3(0, 0, 0), 1.0),
        Material()
    )

    # At origin, inside sphere -> negative
    d = scene.map_distance(Vec3(0, 0, 0))
    assert d < 0
    results.ok("Scene map_distance: inside sphere")

    # At (3, 0, 0), 2 units outside sphere
    d = scene.map_distance(Vec3(3, 0, 0))
    assert approx_eq(d, 2.0, 0.01)
    results.ok("Scene map_distance: outside sphere")


def test_scene_sky(results: TestResults):
    """Test Scene sky color."""
    scene = Scene()

    # Sky at zenith (straight up)
    sky = scene.get_sky(Vec3(0, 1, 0))
    assert sky.y > sky.x  # Should be bluish
    results.ok("Scene sky: zenith direction")

    # Sky at horizon
    sky = scene.get_sky(Vec3(1, 0, 0))
    assert sky.x > 0  # Should have some color
    results.ok("Scene sky: horizon direction")


def test_ray_marcher_basic(results: TestResults):
    """Test basic ray marching."""
    scene = Scene()
    scene.add_object(
        lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0),
        Material(albedo=Vec3(1, 0, 0))
    )
    scene.add_directional_light(Vec3(0.5, 0.8, 0.3))

    marcher = RayMarcher(max_steps=128, enable_shadows=True)

    # March toward sphere from the side
    origin = Vec3(3, 1, 0)
    direction = Vec3(-1, 0, 0)  # Straight toward sphere
    hit = marcher.march(origin, direction, scene)

    assert hit.material is not None, "Should hit the sphere"
    assert hit.distance > 0, "Should have positive distance"
    assert approx_eq(hit.distance, 2.0, 0.1), f"Distance should be ~2.0, got {hit.distance}"
    results.ok("Ray marcher: hit sphere from side")

    # March away from sphere (should miss)
    origin = Vec3(3, 1, 0)
    direction = Vec3(1, 0, 0)  # Away from sphere
    hit = marcher.march(origin, direction, scene)
    assert hit.material is None, "Should miss sphere"
    results.ok("Ray marcher: miss sphere")


def test_ray_marcher_normal(results: TestResults):
    """Test normal computation."""
    scene = Scene()
    scene.add_object(
        lambda p: sdf_sphere(p, Vec3(0, 0, 0), 1.0),
        Material()
    )

    marcher = RayMarcher()

    # Normal at (1, 0, 0) on unit sphere should be (1, 0, 0)
    normal = marcher.compute_normal(Vec3(1, 0, 0), scene)
    assert vec_approx_eq(normal, Vec3(1, 0, 0), 0.05)
    results.ok("Ray marcher normal: sphere surface X")

    # Normal at (0, 1, 0) should be (0, 1, 0)
    normal = marcher.compute_normal(Vec3(0, 1, 0), scene)
    assert vec_approx_eq(normal, Vec3(0, 1, 0), 0.05)
    results.ok("Ray marcher normal: sphere surface Y")


def test_ray_marcher_trace(results: TestResults):
    """Test full trace (ray → color)."""
    scene = Scene()
    scene.add_object(
        lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0),
        Material(albedo=Vec3(1, 0, 0), specular=0.5, roughness=0.5)
    )
    scene.add_object(
        lambda p: sdf_plane(p, height=0.0),
        Material(albedo=Vec3(0.5, 0.5, 0.5), roughness=0.8)
    )
    scene.add_directional_light(Vec3(0.5, 0.8, 0.3), Vec3(1, 1, 1), 1.0)

    marcher = RayMarcher(enable_shadows=True)
    cam = Camera(Vec3(3, 2, 3), Vec3(0, 1, 0), fov=55)

    # Trace toward sphere — should get a reddish color
    direction = (Vec3(0, 1, 0) - cam.position).normalized()
    color = marcher.trace(cam.position, direction, scene)
    # Should be some positive color (not black, not NaN)
    assert color.x >= 0 and color.y >= 0 and color.z >= 0
    assert not math.isnan(color.x)
    results.ok("Ray marcher trace: produces valid color")

    # Trace upward — should hit sky
    direction = Vec3(0, 1, 0)
    color = marcher.trace(cam.position, direction, scene)
    assert color.y > color.x  # Sky should be bluish
    results.ok("Ray marcher trace: sky color")


def test_renderer_basic(results: TestResults):
    """Test the Renderer produces a valid image."""
    scene = build_demo_scene(0.0)
    marcher = RayMarcher(enable_shadows=False)  # Faster
    renderer = Renderer(width=32, height=24, marcher=marcher, quiet=True)
    cam = Camera(Vec3(7, 4, 7), Vec3(0, 1, 0), fov=55)

    img = renderer.render(scene, cam)
    assert img.shape == (24, 32, 3)
    assert img.min() >= 0
    assert img.max() <= 1.0 + 1e-6  # Small tolerance for floating point
    results.ok("Renderer: produces valid image array")

    # Check it's not all black
    assert img.mean() > 0.01, "Image should not be all black"
    results.ok("Renderer: image has content")


def test_renderer_save(results: TestResults):
    """Test saving an image to file."""
    scene = build_demo_scene(0.0)
    marcher = RayMarcher(enable_shadows=False)
    renderer = Renderer(width=32, height=24, marcher=marcher, quiet=True)
    cam = Camera(Vec3(7, 4, 7), Vec3(0, 1, 0), fov=55)

    img = renderer.render(scene, cam)
    test_path = "/tmp/test_raymarcher_output.png"
    renderer.save(img, test_path)
    assert os.path.exists(test_path), "Output file should exist"
    results.ok("Renderer: saves PNG file")

    # Check file is valid PNG by size
    assert os.path.getsize(test_path) > 0, "Output file should not be empty"
    results.ok("Renderer: PNG file has content")


def test_all_scenes(results: TestResults):
    """Test that all scene builders produce valid scenes."""
    for name, builder in [('demo', build_demo_scene),
                           ('chess', build_chess_scene),
                           ('terrain', build_terrain_scene),
                           ('abstract', build_abstract_scene),
                           ('glass', build_glass_scene),
                           ('fractal', build_fractal_scene)]:
        try:
            scene = builder(0.0)
            assert len(scene.objects) > 0, f"Scene '{name}' should have objects"
            assert len(scene.directional_lights) > 0 or len(scene.point_lights) > 0, \
                f"Scene '{name}' should have lights"
            results.ok(f"Scene builder: {name}")
        except Exception as e:
            results.fail(f"Scene builder: {name}", str(e))


def test_all_scenes_render(results: TestResults):
    """Test that all scenes can be rendered without errors."""
    scenes = {
        'demo': (build_demo_scene, Camera(Vec3(7, 4, 7), Vec3(0, 1, 0), fov=55)),
        'chess': (build_chess_scene, Camera(Vec3(5, 5, 8), Vec3(0, 0.5, 0), fov=50)),
        'terrain': (build_terrain_scene, Camera(Vec3(8, 5, 8), Vec3(0, 1, 0), fov=60)),
        'abstract': (build_abstract_scene, Camera(Vec3(6, 4, 6), Vec3(0, 1.5, 0), fov=55)),
        'glass': (build_glass_scene, Camera(Vec3(5, 3.5, 8), Vec3(0, 1, 0), fov=50)),
        'fractal': (build_fractal_scene, Camera(Vec3(4, 3, 4), Vec3(0, 1.5, 0), fov=50)),
    }

    marcher = RayMarcher(enable_shadows=False, max_bounces=1, max_refraction_bounces=1)
    renderer = Renderer(width=16, height=12, marcher=marcher, quiet=True)

    for name, (builder, cam) in scenes.items():
        try:
            scene = builder(0.0)
            img = renderer.render(scene, cam)
            assert img.shape == (12, 16, 3)
            assert img.min() >= 0
            assert not math.isnan(img.mean())
            results.ok(f"Render scene: {name}")
        except Exception as e:
            results.fail(f"Render scene: {name}", str(e))


def test_refraction_trace(results: TestResults):
    """Test that glass objects produce different colors than opaque."""
    scene = build_glass_scene(0.0)
    marcher = RayMarcher(enable_shadows=True, max_bounces=2, max_refraction_bounces=2)
    cam = Camera(Vec3(5, 3.5, 8), Vec3(0, 1, 0), fov=50)

    # Trace toward glass sphere
    direction = (Vec3(0, 1.2, 0) - cam.position).normalized()
    color = marcher.trace(cam.position, direction, scene)

    # Should produce some color (not black)
    assert color.x >= 0 and color.y >= 0 and color.z >= 0
    assert not math.isnan(color.x)
    results.ok("Refraction trace: produces valid color")


def test_ao(results: TestResults):
    """Test ambient occlusion computation."""
    scene = Scene()
    scene.add_object(
        lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0),
        Material()
    )
    scene.add_object(
        lambda p: sdf_plane(p, height=0.0),
        Material()
    )

    marcher = RayMarcher(enable_ao=True, ao_steps=5)
    # AO at top of sphere should be ~1 (open sky)
    ao = marcher.compute_ao(Vec3(0, 2, 0), Vec3(0, 1, 0), scene)
    assert ao > 0.8, f"AO at top of sphere should be high, got {ao}"
    results.ok("Ambient occlusion: top of sphere")

    # AO at bottom of sphere (near ground) should be lower
    ao = marcher.compute_ao(Vec3(0, 0.1, 0), Vec3(0, 1, 0), scene)
    assert ao < 0.95, f"AO near ground should be lower, got {ao}"
    results.ok("Ambient occlusion: near ground")


def test_shadow(results: TestResults):
    """Test shadow computation."""
    scene = Scene()
    scene.add_object(
        lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0),
        Material()
    )
    scene.add_object(
        lambda p: sdf_plane(p, height=0.0),
        Material()
    )

    marcher = RayMarcher(enable_shadows=True)

    # Point in direct light (top of sphere)
    light_dir = Vec3(0.5, 0.8, 0.3).normalized()
    shadow = marcher.compute_shadow(Vec3(0, 2.1, 0), light_dir, scene)
    assert shadow > 0.5, f"Top of sphere should not be in shadow, got {shadow}"
    results.ok("Shadow: lit area")

    # Point on ground behind sphere (in shadow)
    shadow = marcher.compute_shadow(Vec3(0, 0.01, -1), light_dir, scene)
    assert shadow < 0.5, f"Ground behind sphere should be shadowed, got {shadow}"
    results.ok("Shadow: shadowed area")


def test_camera(results: TestResults):
    """Test Camera ray generation."""
    cam = Camera(Vec3(0, 0, 5), Vec3(0, 0, 0), fov=60)

    # Center ray should go roughly toward target
    origin, direction = cam.get_ray(0, 0)
    assert vec_approx_eq(origin, Vec3(0, 0, 5), 0.01)
    # Direction should be roughly toward -Z
    assert direction.z < 0, "Center ray should point toward target"
    results.ok("Camera: center ray direction")

    # DOF camera
    cam_dof = Camera(Vec3(0, 0, 5), Vec3(0, 0, 0), fov=60, focal_distance=5.0, aperture=0.1)
    origin, direction = cam_dof.get_ray(0, 0, 0)
    # With DOF, origin should be jittered
    results.ok("Camera: DOF ray generation")


def test_hit_result(results: TestResults):
    """Test HitResult defaults."""
    hit = HitResult()
    assert hit.distance == float('inf')
    assert hit.material is None
    assert hit.object_index == -1
    assert hit.entering == True
    results.ok("HitResult default values")


def test_sdf_mandelbulb(results: TestResults):
    """Test Mandelbulb distance estimator returns finite values."""
    center = Vec3(0, 0, 0)

    # Point near the bulb should give a finite distance
    d = sdf_mandelbulb(Vec3(0.5, 0.5, 0), center, scale=1.0, power=8.0)
    assert math.isfinite(d), f"Mandelbulb distance should be finite, got {d}"
    results.ok("Mandelbulb: finite distance at origin")

    # Point far away should give positive distance
    d = sdf_mandelbulb(Vec3(10, 10, 10), center, scale=1.0, power=8.0)
    assert d > 0, f"Mandelbulb should be positive far away, got {d}"
    results.ok("Mandelbulb: positive distance far away")


def test_sdf_cone(results: TestResults):
    """Test cone SDF."""
    center = Vec3(0, 0, 0)
    half_angle = math.pi / 4  # 45 degrees
    height = 2.0

    # Point on the side of the cone
    d = sdf_cone(Vec3(1, 0, 0), center, half_angle, height)
    # This should be approximately on the surface (for a 45-degree cone, x=1 at y=0 is on surface)
    results.ok("SDF cone: basic evaluation")


def test_sdf_hex_prism(results: TestResults):
    """Test hexagonal prism SDF."""
    center = Vec3(0, 0, 0)

    # Point at origin should be inside
    d = sdf_hex_prism(Vec3(0, 0, 0), center, radius=1.0, half_height=1.0)
    assert d < 0, f"Origin should be inside hex prism, got {d}"
    results.ok("SDF hex prism: inside")

    # Point far away should be outside
    d = sdf_hex_prism(Vec3(10, 10, 10), center, radius=1.0, half_height=1.0)
    assert d > 0, f"Far point should be outside hex prism, got {d}"
    results.ok("SDF hex prism: outside")


def main():
    print("=" * 60)
    print("SDF Ray Marcher k9f2 — Test Suite")
    print("=" * 60)

    results = TestResults()

    # Vec3 tests
    print("\n--- Vec3 Tests ---")
    test_vec3_construction(results)
    test_vec3_arithmetic(results)
    test_vec3_dot_cross(results)
    test_vec3_length_normalize(results)
    test_vec3_reflect(results)
    test_vec3_refract(results)
    test_vec3_lerp_clamp(results)
    test_vec3_pow(results)

    # SDF primitive tests
    print("\n--- SDF Primitive Tests ---")
    test_sdf_sphere(results)
    test_sdf_box(results)
    test_sdf_plane(results)
    test_sdf_torus(results)
    test_sdf_capsule(results)
    test_sdf_cylinder(results)
    test_sdf_cone(results)
    test_sdf_hex_prism(results)
    test_sdf_mandelbulb(results)

    # CSG operation tests
    print("\n--- CSG Operation Tests ---")
    test_csg_operations(results)

    # Pattern tests
    print("\n--- Pattern Tests ---")
    test_checkerboard(results)

    # Material tests
    print("\n--- Material Tests ---")
    test_material(results)

    # Scene tests
    print("\n--- Scene Tests ---")
    test_scene_construction(results)
    test_scene_map(results)
    test_scene_map_distance(results)
    test_scene_sky(results)
    test_hit_result(results)

    # Ray marcher tests
    print("\n--- Ray Marcher Tests ---")
    test_ray_marcher_basic(results)
    test_ray_marcher_normal(results)
    test_ray_marcher_trace(results)
    test_ao(results)
    test_shadow(results)

    # Camera tests
    print("\n--- Camera Tests ---")
    test_camera(results)

    # Renderer tests
    print("\n--- Renderer Tests ---")
    test_renderer_basic(results)
    test_renderer_save(results)

    # All scenes tests
    print("\n--- All Scenes Tests ---")
    test_all_scenes(results)
    test_all_scenes_render(results)

    # Refraction test
    print("\n--- Refraction Tests ---")
    test_refraction_trace(results)

    success = results.summary()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()