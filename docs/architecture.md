# Architecture Guide

This document provides a deep dive into the architecture and rendering pipeline
of the SDF Ray Marcher k9f2.

## Overview

The SDF Ray Marcher renders 3D scenes using **signed distance fields** (SDFs) —
mathematical functions that return the shortest distance from any point in space
to the surface of a shape. Negative values mean "inside the shape," zero means
"on the surface," and positive means "outside."

This is fundamentally different from triangle-based renderers (like OpenGL):
instead of sending triangles to the GPU, we evaluate mathematical functions
for every pixel on screen.

## Core Pipeline

```
┌─────────────┐
│   Camera    │ ─── For each pixel, generate a ray (origin + direction)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  RayMarcher │ ─── March ray through scene using SDF distance
│   .trace()  │     └── For each step:
└──────┬──────┘         1. Evaluate Scene.map() to find closest distance
       │                2. If distance < ε → hit! Compute shading
       │                3. Otherwise, advance ray by that distance
       │
       ▼
┌─────────────┐
│   Shading   │ ─── Compute surface color:
│   .shade()  │     ├── Ambient + AO
└──────┬──────┘     ├── Diffuse + Specular (Blinn-Phong)
       │            ├── Soft/Hard shadows
       │            ├── Subsurface scattering
       │            ├── Reflections (recursive trace)
       │            └── Refractions (recursive trace)
       │
       ▼
┌─────────────┐
│   Renderer  │ ─── Assemble all pixels into a NumPy array
│  .render()  │     Apply exposure + gamma correction
└──────┬──────┘     Optional tone mapping
       │
       ▼
┌─────────────┐
│    save()   │ ─── Convert float array → uint8 PNG via Pillow
└─────────────┘
```

## Module Breakdown

### `vec3.py` — 3D Vector Math

The `Vec3` class provides all the 3D math needed for ray marching:
- Arithmetic: `+`, `-`, `*` (scalar and component-wise), negation
- Products: `dot()`, `cross()`
- Geometry: `length()`, `length_sq()`, `normalized()`, `reflect()`, `refract()`
- Utilities: `lerp()`, `clamp()`, `pow()`, `to_tuple()`, `from_tuple()`

### `primitives.py` — SDF Functions

Each SDF function takes a point `p` and shape parameters, returning a float:

| SDF | Inside | On surface | Outside |
|-----|--------|------------|---------|
| `sdf_sphere(p, center, r)` | < 0 | = 0 | > 0 |
| `sdf_box(p, center, half)` | < 0 | = 0 | > 0 |
| `sdf_torus(p, center, R, r)` | < 0 | = 0 | > 0 |
| ... | ... | ... | ... |

The key property: the SDF gradient (estimated via central differences) gives
the **surface normal** at any point.

### `transforms.py` — SDF Transformations

Transforms modify points before passing them to SDF functions, enabling
scene graphs with positioned, rotated, and scaled objects:

- `sdf_translate(p, offset, sdf_func)` — Move an object
- `sdf_scale(p, scale, sdf_func)` — Uniformly scale an object
- `sdf_rotate_x/y/z(p, angle, sdf_func)` — Rotate around an axis
- `sdf_repeat(p, period, sdf_func)` — Infinite repetition
- `sdf_elongate(p, elongation, sdf_func)` — Elongate along axes

### `csg.py` — Constructive Solid Geometry

CSG operations combine SDFs to create complex shapes from simple ones:

- `sdf_union(d1, d2)` — `min(d1, d2)` — Boolean OR
- `sdf_smooth_union(d1, d2, k)` — Blended union (organic shapes)
- `sdf_subtraction(d1, d2)` — `max(-d2, d1)` — Boolean AND NOT
- `sdf_intersection(d1, d2)` — `max(d1, d2)` — Boolean AND

### `material.py` — Surface Properties

Materials control how light interacts with a surface:

| Property | Effect |
|----------|--------|
| `albedo` | Base diffuse color |
| `specular` | Specular highlight intensity |
| `roughness` | 0 = mirror reflection, 1 = diffuse |
| `reflectivity` | Mirror reflection strength |
| `ior` | Index of refraction (1.0=air, 1.33=water, 1.5=glass) |
| `transparency` | Glass-like transparency |
| `subsurface` | Subsurface scattering strength |
| `emissive` | Self-illumination (bypasses lighting) |

Presets: `glass()`, `gold()`, `mirror()`, `water()`, `copper()`, `jade()`,
`chrome()`, `marble()`, `emissive()`, `checkerboard()`

### `marcher.py` — Ray Marching Engine

The core algorithm:

```python
def march(origin, direction, scene):
    t = 0.0
    for step in range(max_steps):
        p = origin + direction * t
        d = scene.map_distance(p)  # Get closest SDF distance
        if d < epsilon:            # Close enough → hit
            return HitResult(distance=t, ...)
        t += d                      # Advance by closest distance
        if t > max_distance:       # Too far → miss
            break
    return HitResult()              # No hit
```

Key features:
- **Shadow rays**: March toward the light; if blocked → shadow
- **Soft shadows**: Use penumbra ratio (closest approach / distance) for soft edges
- **Ambient occlusion**: Sample SDF along normal; concave areas get darker
- **Reflections**: Recursive `trace()` with Fresnel blending
- **Refractions**: Snell's law with proper IOR handling, TIR support

### `scene.py` — Scene Container

A `Scene` holds:
- Objects: list of `(sdf_func, material)` pairs
- Lights: `DirectionalLight` and `PointLight`
- Environment: `sky_color`, `fog_color`, `fog_density`, `background_func`

The `map()` method evaluates all SDFs and returns the closest one.

### `camera.py` — Perspective Camera

Thin-lens model with:
- Look-at targeting (position → target)
- Vertical FOV
- Depth-of-field via aperture sampling (golden angle distribution)

### `renderer.py` — Image Assembler

- Iterates all pixels
- Generates rays via `Camera.get_ray()`
- Traces each ray via `RayMarcher.trace()`
- Applies exposure and gamma correction
- Outputs NumPy array or saves as PNG

### `vec_renderer.py` — Multiprocessing Renderer

`MultiprocessRenderer` splits the image into horizontal bands and renders
each in a separate process for near-linear speedup on multi-core machines.

### `tonemap.py` — Tone Mapping Operators

HDR-to-LDR conversion operators:
- **ACES** (default): Industry standard filmic curve
- **Reinhard**: `x / (1+x)` — simple, washes out brights
- **Filmic**: Uncharted 2 curve — cinematic look
- **Clamped**: Hard clamp to [0, 1]
- **Linear**: No mapping (pass-through)

### `config.py` — YAML/TOML Configuration

Declares primitives, materials, lights, camera, and marcher settings in
YAML or TOML files for no-code scene creation.

### `cli.py` — Command-Line Interface

Full argparse CLI with all rendering options, tone mapping, multiprocessing,
animation, and scene listing.

## Data Flow Example

Rendering a single pixel:

```
1. Camera.get_ray(u=0.3, v=0.2, aspect=1.33, sample=0)
   → (origin=Vec3(7,4,7), direction=normalized(target - origin + offset))

2. RayMarcher.trace(origin, direction, scene)
   ├── RayMarcher.march() → HitResult(distance=5.3, material=red_sphere)
   │       └── Scene.map(Vec3(1.7, 3.2, 4.1)) → closest SDF = 0.001
   │
   ├── RayMarcher.compute_normal(Vec3(1.7, 3.2, 4.1), scene) → Vec3(0.70, -0.71, 0.0)
   │
   ├── RayMarcher.shade() → Vec3(0.85, 0.18, 0.12)  # Red with lighting
   │       ├── Ambient: Vec3(0.12, 0.03, 0.02)
   │       ├── Diffuse:  Vec3(0.65, 0.14, 0.09)
   │       └── Specular: Vec3(0.08, 0.08, 0.08)  # White highlight
   │
   ├── Reflection: trace(reflect_dir, scene, bounce=1) → Vec3(0.05, 0.05, 0.06)
   │
   └── Final: mix(shade, reflection, reflectivity) → Vec3(0.73, 0.16, 0.11)

3. Apply exposure (×1.0) and gamma (^(1/2.2))
   → Vec3(0.88, 0.41, 0.34)  →  img[y, x] = [0.88, 0.41, 0.34]

4. After all pixels: convert to uint8 PNG
   → [224, 104, 87]  →  saved to output.png
```

## Performance Characteristics

Pure-Python means:
- ✅ Portable: no GPU, no C extensions needed
- ✅ Readable: every algorithm is explicit Python
- ✅ Hackable: easy to modify and extend
- ❌ Slow: ~1000× slower than GPU renderers
- ❌ Single-threaded by default (use `--multiprocess` for parallel)

Typical rendering times:

| Resolution | Scene | Single | Multiprocess (8 cores) |
|-----------|-------|--------|------------------------|
| 320×240 | demo | ~5s | ~1s |
| 800×600 | demo | ~45s | ~8s |
| 800×600 | glass | ~120s | ~20s |

## Extending the Renderer

### Adding a new SDF primitive

1. Add the SDF function to `primitives.py`
2. Add it to `PRIMITIVE_MAP` in `config.py`
3. Add it to `__init__.py` exports
4. Write tests in `tests/test_raymarcher.py`
5. Update the README primitives table

### Adding a new material preset

1. Add the factory function to `material.py`
2. Export from `__init__.py`
3. Add a test

### Adding a new scene

1. Add the builder function to `scenes.py`
2. Register in `SCENES` dict
3. Add camera preset in `cli.py`
4. Add tests that it builds and renders