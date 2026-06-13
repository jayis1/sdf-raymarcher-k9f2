# SDF Ray Marcher k9f2

A pure-Python signed distance field (SDF) ray marcher that renders 3D scenes with physically-based lighting, reflections, refractions (glass), subsurface scattering, soft shadows, ambient occlusion, depth-of-field, and procedural fractals — all without any GPU or graphics API.

## How It Works

The renderer uses **ray marching**: for each pixel, a ray is cast from the camera through the scene. At each step, the ray advances by the distance to the closest surface (the signed distance field value). This continues until the ray either hits a surface (distance < epsilon) or escapes to infinity.

### Key Concepts

- **Signed Distance Fields (SDFs)**: Each object is defined by a mathematical function that returns the shortest distance from any point in space to the surface. Negative values mean "inside" the object.
- **Ray Marching**: Rays advance through the scene in adaptive steps, with step size determined by the SDF value — fast in open space, slow near surfaces.
- **Normal Estimation**: Surface normals are computed via central finite differences of the SDF gradient.
- **CSG Operations**: Boolean union, subtraction, intersection, and smooth blending allow composing complex shapes from primitives.
- **Physically-Based Shading**: Blinn-Phong specular, Lambert diffuse, Schlick Fresnel for reflections.
- **Refraction**: Snell's law with total internal reflection, Beer's law absorption for colored glass.
- **Subsurface Scattering**: Approximation via back-light transmission for wax/skin-like materials.
- **Soft Shadows**: By tracking closest approach distance along shadow rays, producing realistic penumbra.
- **Ambient Occlusion**: Estimated by sampling the SDF along the surface normal to detect nearby geometry.
- **Depth of Field**: Thin-lens camera model with disk-sampled aperture for bokeh effects.
- **Procedural Patterns**: Checkerboard materials applied per-pixel without textures.

### Supported Primitives

| Primitive | Description |
|-----------|-------------|
| Sphere | Classic SDF sphere |
| Box | Axis-aligned box |
| Torus | Ring-shaped torus |
| Cylinder | Vertical cylinder |
| Capsule | Line-segment swept sphere |
| Plane | Infinite ground plane |
| Cone | Upward-pointing cone |
| Hex Prism | Hexagonal prism |
| Mandelbulb | Fractal distance estimator |

### CSG Operations

- `sdf_union` — Boolean union (A ∪ B)
- `sdf_smooth_union` — Smooth blended union with configurable radius
- `sdf_smooth_subtraction` — Smooth blended subtraction
- `sdf_subtraction` — Boolean subtraction (A − B)
- `sdf_intersection` — Boolean intersection (A ∩ B)

### Material Properties

| Property | Description |
|----------|-------------|
| `albedo` | Base diffuse color |
| `specular` | Specular highlight strength (0–1) |
| `roughness` | 0 = mirror, 1 = fully diffuse |
| `reflectivity` | 0 = no reflection, 1 = perfect mirror |
| `emissive` | Self-illumination color (bypasses shading) |
| `fresnel` | Schlick Fresnel base reflectivity |
| `ior` | Index of refraction (1.0=air, 1.33=water, 1.5=glass) |
| `transparency` | 0 = opaque, 1 = fully transparent (enables refraction) |
| `subsurface` | Subsurface scattering strength |
| `subsurface_color` | SSS tint color |
| `checker_scale` | If > 0, applies procedural checkerboard pattern |
| `checker_color` | Secondary checkerboard color |

## Usage

### Setup

```bash
cd ~/projects/sdf-raymarcher-k9f2
source venv/bin/activate
```

### Render a scene

```bash
# Default demo scene
python3 raymarcher.py

# High-quality render with AO and soft shadows
python3 raymarcher.py --ao --soft-shadow --samples 4

# Chess scene at higher resolution
python3 raymarcher.py --scene chess --width 1280 --height 960 --output chess.png

# Glass scene (showcasing refraction)
python3 raymarcher.py --scene glass --width 800 --height 600

# Fractal scene (Mandelbulb)
python3 raymarcher.py --scene fractal --width 640 --height 480

# With depth of field
python3 raymarcher.py --scene demo --dof 10 --aperture 0.15

# Abstract animation (8 frames)
python3 raymarcher.py --scene abstract --animate 8

# Terrain with no shadows (faster)
python3 raymarcher.py --scene terrain --no-shadow --output terrain.png

# Quiet mode (suppress progress)
python3 raymarcher.py --scene demo --quiet --output render.png
```

### All Options

```
  --width W          Image width (default: 800)
  --height H         Image height (default: 600)
  --samples N        Samples per pixel for anti-aliasing (default: 1)
  --output PATH      Output file path (default: output.png)
  --scene NAME       Scene: demo, chess, terrain, abstract, glass, fractal (default: demo)
  --animate N        Render N animation frames (default: 1, single frame)
  --max-bounces N    Max reflection/refraction bounces (default: 3)
  --ao               Enable ambient occlusion (slower)
  --soft-shadow      Enable soft shadows (slower)
  --no-shadow        Disable shadows entirely
  --dof DIST         Depth-of-field focal distance (0 = disabled)
  --aperture FLOAT   DOF aperture size (default: 0.1)
  --quiet            Suppress progress output
  --steps N          Max ray march steps per ray (default: 128)
```

### Built-in Scenes

- **demo** — Various primitives (sphere, box, torus, capsule, smooth union, cylinder) on a reflective checker ground
- **chess** — A chess board with king and pawn pieces, atmospheric fog
- **terrain** — Procedural terrain with animated water plane and fog
- **abstract** — Morphing smooth-union blob animation
- **glass** — Refractive glass spheres/cubes, subsurface scattering, water sphere
- **fractal** — Animated Mandelbulb fractal with varying power parameter

### Programmatic Usage

```python
from raymarcher import (
    Scene, Camera, Renderer, RayMarcher, Material, Vec3,
    sdf_sphere, sdf_smooth_union
)

scene = Scene()
scene.add_object(
    lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0),
    Material(albedo=Vec3(1, 0, 0), reflectivity=0.5)
)
scene.add_directional_light(Vec3(0.5, 0.8, 0.3))

# Glass material example
glass = Material(
    albedo=Vec3(0.95, 0.95, 0.98),
    transparency=0.9, ior=1.5, fresnel=0.04,
    specular=0.9, roughness=0.02,
)
scene.add_object(
    lambda p: sdf_sphere(p, Vec3(3, 1, 0), 1.0),
    glass
)

marcher = RayMarcher(enable_shadows=True, enable_ao=True, max_bounces=3)
renderer = Renderer(width=640, height=480, marcher=marcher)

# Camera with depth of field
camera = Camera(
    Vec3(5, 3, 5), Vec3(0, 1, 0), fov=55,
    focal_distance=8.0, aperture=0.1
)

img = renderer.render(scene, camera)
renderer.save(img, 'my_scene.png')
```

## Architecture

```
raymarcher.py
├── Vec3              — 3D vector math (add, sub, mul, dot, cross, reflect, refract, pow, ...)
├── Material           — Surface properties (albedo, IOR, transparency, SSS, checkerboard, ...)
├── SDF Primitives    — sphere, box, torus, cylinder, capsule, plane, cone, hex_prism, mandelbulb
├── CSG Operations    — union, smooth_union, smooth_subtraction, subtraction, intersection
├── Procedural Patterns — checkerboard
├── SceneObject        — SDF function + material + optional interior material
├── Scene              — Object collection, lights, sky evaluation, fog
├── RayMarcher         — Core engine (march, normal, AO, shadow, SSS, shade, trace with refraction)
├── Camera             — Perspective camera with DOF (thin lens model)
└── Renderer           — Pixel loop, anti-aliasing, gamma correction, PNG output

test_raymarcher.py
└── 90 tests covering Vec3 math, SDF primitives, CSG ops, materials, scenes, rendering

test_bugs.py
└── 19 bug-specific tests covering FOV, entering detection, refraction IOR, cone SDF, aspect ratio, etc.
```

## Testing

```bash
source venv/bin/activate
python3 test_raymarcher.py
```

The test suite covers:
- Vec3 arithmetic, normalization, reflection, refraction, lerp, clamp, pow
- All SDF primitives (surface, inside, outside, distance accuracy)
- CSG operations (union, smooth union, subtraction, intersection)
- Checkerboard pattern
- Material properties
- Scene construction, map, map_distance
- Ray marching (hit detection, normal computation, shadow, AO)
- Camera ray generation including DOF
- All 6 scene builders
- Full rendering pipeline for all scenes
- Refraction trace validation

## Requirements

- Python 3.11+
- NumPy
- Pillow

## Known Issues (Resolved)

The following bugs were found and fixed during the Phase 3 bug hunt:

1. **Camera FOV double-scaling (Critical)**: `Renderer.render()` pre-scaled `u,v` coordinates by `half_w/half_h` and then `Camera.get_ray()` applied FOV scaling again internally, effectively doubling the FOV. This made rendered images appear zoomed in compared to the specified FOV. **Fix**: Removed pre-scaling from `render()`, changed `get_ray()` to accept an `aspect` parameter, and pass normalized `[-0.5, 0.5]` coordinates directly.

2. **Ray march entering detection (Critical)**: `march()` determined whether a ray was entering or exiting an object based on the SDF value at the hit point, which is always near `surface_epsilon` (ambiguous). This caused refraction to use the wrong IOR when exiting glass objects. **Fix**: Check the SDF at the ray *origin* instead — if the origin is inside an object (SDF < 0), the ray is exiting.

3. **Refraction IOR flip bug (Critical)**: When exiting a glass object, `eta` was set to `1/ior` (same as entering) instead of `ior`. Per Snell's law, `eta = n1/n2`: entering is `1.0/1.5 ≈ 0.667` (air→glass), but exiting should be `1.5` (glass→air). **Fix**: Set `eta = ior` when exiting.

4. **Cone SDF incorrect (Moderate)**: The cone SDF didn't properly cap the base or handle inside/outside regions. Points below the base got wrong distances, and the formula didn't match the standard capped cone SDF. **Fix**: Rewrote with proper base capping using the `max(cone_dist, base_dist)` intersection.

5. **Hex prism dead code (Minor)**: `sdf_hex_prism()` computed `d_xz` on line 259 but then immediately overwrote it with `d_hex` computed separately. The initial computation and misleading comment were removed.

6. **Dead code in refraction IOR logic (Minor)**: Lines 707-708 assigned `eta = ior` and then immediately overwrote with `eta = 1.0 / ior` with a comment explaining the correction. This was confusing dead code. **Fix**: Removed the dead assignment and clarified the comment.

7. **Renderer dead code (Minor)**: `render()` computed `fov_rad`, `half_h`, `half_w`, `forward`, `right`, `true_up` but never used them — these were leftover from when rendering was done directly in the method. **Fix**: Removed the unused computations.

## License

MIT