# SDF Ray Marcher k9f2

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-109%2B-green.svg)](tests/)

A pure-Python signed distance field (SDF) ray marcher that renders 3D scenes with
physically-based lighting, reflections, refractions (glass), subsurface scattering,
soft shadows, ambient occlusion, depth-of-field, animation, and procedural fractals
— all without any GPU or graphics API.

```
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │   ╔════╗          ┌──┐                                   │
    │   ║    ║   ┌──────┘  └──────┐   ◉ Glass sphere          │
    │   ║    ║   │   Torus        │   ● Reflective sphere      │
    │   ╚════╝   └───────────────┘   ■ Box with shadows         │
    │                                                          │
    │   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
    │   ░░░░░░░░░░░░ Checkerboard Ground ░░░░░░░░░░░░░░░░░░  │
    └──────────────────────────────────────────────────────────┘
```

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Programmatic Usage](#programmatic-usage)
- [Scene Configuration](#scene-configuration)
- [Architecture](#architecture)
- [Primitives Reference](#primitives-reference)
- [CSG Operations](#csg-operations)
- [Material Properties](#material-properties)
- [Material Presets](#material-presets)
- [Built-in Scenes](#built-in-scenes)
- [Testing](#testing)
- [Performance](#performance)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Changelog](#changelog)

## Features

- **Pure Python** — no GPU, no C extensions, no OpenGL. Just NumPy + Pillow.
- **Physically-based shading** — Blinn-Phong specular, Lambert diffuse, Schlick Fresnel
- **Refractions** — Snell's law with total internal reflection, Beer's law absorption
- **Reflections** — Recursive mirror reflections with Fresnel blending
- **Subsurface scattering** — Approximated via back-light transmission
- **Soft shadows** — Penumbra via closest approach distance
- **Ambient occlusion** — Estimated by sampling SDF along surface normals
- **Depth of field** — Thin-lens camera model with disk-sampled aperture
- **9 SDF primitives** — Sphere, box, torus, cylinder, capsule, plane, cone, hex prism, Mandelbulb
- **CSG operations** — Union, smooth union, subtraction, intersection, smooth variants
- **Procedural patterns** — Checkerboard, stripes, gradient
- **7 built-in scenes** — Demo, chess, terrain, abstract, glass, fractal, studio
- **YAML/TOML config** — Define scenes declaratively without writing Python
- **CLI interface** — Render from the command line with full control
- **Animation support** — Render frame sequences for rotating cameras
- **Installable package** — `pip install -e .` for development
- **Comprehensive tests** — 109+ tests covering all modules

## Installation

### From source (recommended)

```bash
git clone https://github.com/jayis1/sdf-raymarcher-k9f2.git
cd sdf-raymarcher-k9f2
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### Dependencies

- Python 3.11+
- NumPy
- Pillow
- PyYAML (for config files, optional)
- toml (for TOML config, optional)
- pytest (for development)

## Quick Start

### Command Line

```bash
# Render the demo scene
raymarcher-k9f2 --scene demo --output demo.png

# High-quality render with AO and soft shadows
raymarcher-k9f2 --scene glass --ao --soft-shadow --samples 4 --output glass_hq.png

# Fractal scene at high resolution
raymarcher-k9f2 --scene fractal --width 1280 --height 960 --output fractal.png

# Depth of field
raymarcher-k9f2 --scene demo --dof 10 --aperture 0.15 --output dof.png

# Animation (8 frames)
raymarcher-k9f2 --scene abstract --animate 8 --output frame.png

# From config file
raymarcher-k9f2 --config examples/scene.yaml --output custom.png

# List available scenes
raymarcher-k9f2 --list-scenes

# Version info
raymarcher-k9f2 --version
```

### Programmatic

```python
from raymarcher_k9f2 import (
    Scene, Camera, Renderer, RayMarcher, Material, Vec3,
    sdf_sphere, sdf_plane,
)

scene = Scene()

# Add a red sphere
scene.add_object(
    lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0),
    Material(albedo=Vec3(1, 0, 0), reflectivity=0.5)
)

# Add a ground plane
scene.add_object(
    lambda p: sdf_plane(p, height=0.0),
    Material(albedo=Vec3(0.5, 0.5, 0.5), checker_scale=1.0)
)

# Add lighting
scene.add_directional_light(Vec3(0.5, 0.8, 0.3))

# Render
marcher = RayMarcher(enable_shadows=True, enable_ao=True, max_bounces=3)
renderer = Renderer(width=640, height=480, marcher=marcher)
camera = Camera(Vec3(5, 3, 5), Vec3(0, 1, 0), fov=55, focal_distance=8.0, aperture=0.1)

img = renderer.render(scene, camera)
renderer.save(img, 'my_scene.png')

# Or get a PIL Image directly
pil_img = renderer.render_to_pil(scene, camera)
pil_img.show()
```

### Using Material Presets

```python
from raymarcher_k9f2 import Scene, Vec3, sdf_sphere
from raymarcher_k9f2.material import glass, gold, mirror, water, checkerboard

scene = Scene()

# Glass sphere (IOR 1.5)
scene.add_object(lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0), glass())

# Gold sphere
scene.add_object(lambda p: sdf_sphere(p, Vec3(3, 1, 0), 1.0), gold())

# Mirror sphere
scene.add_object(lambda p: sdf_sphere(p, Vec3(-3, 1, 0), 1.0), mirror())

# Checkerboard ground
scene.add_object(lambda p: sdf_sphere(p, Vec3(0, -0.01, 0), 100.0), checkerboard(scale=2.0))
```

## CLI Reference

```
usage: raymarcher-k9f2 [-h] [--width W] [--height H] [--samples N]
                       [--output PATH] [--scene NAME] [--config FILE]
                       [--animate N] [--max-bounces N] [--ao]
                       [--soft-shadow] [--no-shadow] [--dof DIST]
                       [--aperture FLOAT] [--gamma FLOAT] [--exposure FLOAT]
                       [--quiet] [--verbose] [--steps N] [--list-scenes]
                       [--version]

Options:
  --width W          Image width (default: 800)
  --height H         Image height (default: 600)
  --samples N        Samples per pixel for anti-aliasing (default: 1)
  --output PATH      Output file path (default: output.png)
  --scene NAME       Built-in scene: demo, chess, terrain, abstract,
                     glass, fractal, studio (default: demo)
  --config FILE      Path to YAML/TOML scene config (overrides --scene)
  --animate N        Number of animation frames (default: 1)
  --max-bounces N    Max reflection/refraction bounces (default: 3)
  --ao               Enable ambient occlusion
  --soft-shadow      Enable soft shadows
  --no-shadow        Disable shadows entirely
  --dof DIST         Depth-of-field focal distance (0 = disabled)
  --aperture FLOAT   DOF aperture size (default: 0.1)
  --gamma FLOAT      Gamma correction exponent (default: 2.2)
  --exposure FLOAT   Linear exposure multiplier (default: 1.0)
  --quiet            Suppress progress output
  --verbose          Enable debug logging
  --steps N          Max ray march steps per ray (default: 128)
  --list-scenes      List available scenes and exit
  --version          Print version and exit
```

## Scene Configuration

Define scenes declaratively using YAML or TOML config files:

```yaml
scene:
  width: 800
  height: 600
  output: my_scene.png
  camera:
    position: [7, 4, 7]
    target: [0, 1, 0]
    fov: 55
  marcher:
    max_steps: 128
    max_bounces: 3
    enable_shadows: true
  objects:
    - type: sphere
      center: [0, 1, 0]
      radius: 1.0
      material:
        albedo: [0.9, 0.2, 0.15]
        specular: 0.9
        roughness: 0.05
        reflectivity: 0.4
    - type: plane
      height: 0.0
      material:
        albedo: [0.7, 0.7, 0.7]
        checker_scale: 1.0
  lights:
    - type: directional
      direction: [0.5, 0.8, 0.3]
      color: [1.0, 0.95, 0.85]
      intensity: 1.0
```

Render with: `raymarcher-k9f2 --config examples/scene.yaml`

Supported object types: `sphere`, `box`, `torus`, `cylinder`, `capsule`, `plane`, `cone`.

## Architecture

```
src/raymarcher_k9f2/
├── __init__.py          — Package entry point, public API exports
├── vec3.py              — Vec3 class: 3D vector math (add, dot, cross, reflect, refract, ...)
├── material.py          — Material dataclass + presets (glass, gold, mirror, water, ...)
├── primitives.py         — SDF functions (sphere, box, torus, cylinder, capsule, plane, cone, hex_prism, mandelbulb)
├── csg.py               — CSG operations (union, smooth_union, subtraction, intersection, ...)
├── patterns.py           — Procedural patterns (checkerboard, stripes, gradient)
├── scene.py             — Scene, SceneObject, PointLight, DirectionalLight, HitResult
├── marcher.py            — RayMarcher: core engine (march, normal, AO, shadow, shade, trace)
├── camera.py             — Camera: perspective camera with thin-lens DOF
├── renderer.py           — Renderer: pixel loop, AA, gamma, save, render_to_pil
├── scenes.py             — Built-in scene builders (demo, chess, terrain, abstract, glass, fractal, studio)
├── config.py             — YAML/TOML config loader and scene builder
└── cli.py                — CLI entry point (argparse)
tests/
└── test_raymarcher.py   — 109+ pytest-compatible tests
examples/
├── scene.yaml            — Example YAML config
├── simple_sphere.py      — Simple sphere example
└── glass_scene.py        — Glass/refraction example
```

### Data Flow

```
Camera.get_ray(u, v) → (origin, direction)
        │
        ▼
RayMarcher.trace(origin, direction, scene)
        │
        ├── RayMarcher.march() → HitResult (distance, material, entering)
        │       └── Scene.map() → closest SDF distance + material
        │
        ├── RayMarcher.compute_normal() → surface normal
        │
        ├── RayMarcher.shade() → color with lighting
        │       ├── Ambient + optional AO
        │       ├── Diffuse + Specular (Blinn-Phong)
        │       ├── Soft/Hard shadows
        │       └── Subsurface scattering
        │
        ├── Reflection: trace(reflect_dir) → blended via Fresnel
        │
        ├── Refraction: trace(refract_dir) → blended via Beer's law
        │
        └── Fog → final color
                │
                ▼
Renderer.render() → numpy array → save() → PNG
```

## Primitives Reference

| Primitive | Function | Parameters |
|-----------|----------|------------|
| Sphere | `sdf_sphere(p, center, radius)` | center, radius |
| Box | `sdf_box(p, center, half_extents)` | center, half_extents (Vec3) |
| Torus | `sdf_torus(p, center, major_r, minor_r)` | center, major_r, minor_r |
| Cylinder | `sdf_cylinder(p, center, radius, half_height)` | center, radius, half_height |
| Capsule | `sdf_capsule(p, a, b, radius)` | endpoints a, b, radius |
| Plane | `sdf_plane(p, height)` | height (Y position) |
| Cone | `sdf_cone(p, center, half_angle, height)` | center, half_angle (radians), height |
| Hex Prism | `sdf_hex_prism(p, center, radius, half_height)` | center, radius, half_height |
| Mandelbulb | `sdf_mandelbulb(p, center, scale, power, iterations)` | center, scale, power, iterations |

## CSG Operations

| Operation | Function | Description |
|-----------|----------|-------------|
| Union | `sdf_union(d1, d2)` | Boolean union (min) |
| Smooth Union | `sdf_smooth_union(d1, d2, k)` | Blended union with radius k |
| Subtraction | `sdf_subtraction(d1, d2)` | Boolean subtraction (A - B) |
| Smooth Subtraction | `sdf_smooth_subtraction(d1, d2, k)` | Blended subtraction |
| Intersection | `sdf_intersection(d1, d2)` | Boolean intersection (max) |
| Smooth Intersection | `sdf_smooth_intersection(d1, d2, k)` | Blended intersection |

## Material Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `albedo` | Vec3 | (0.8, 0.8, 0.8) | Base diffuse color |
| `specular` | float | 0.5 | Specular highlight strength (0–1) |
| `roughness` | float | 0.3 | 0 = mirror, 1 = fully diffuse |
| `reflectivity` | float | 0.0 | Mirror reflection strength (0–1) |
| `emissive` | Vec3 | (0, 0, 0) | Self-illumination color |
| `fresnel` | float | 0.04 | Schlick Fresnel base reflectivity |
| `ior` | float | 1.5 | Index of refraction |
| `transparency` | float | 0.0 | Transparency (0=opaque, 1=glass) |
| `subsurface` | float | 0.0 | Subsurface scattering strength |
| `subsurface_color` | Vec3 | (0.5, 0.2, 0.1) | SSS tint color |
| `checker_scale` | float | 0.0 | Checkerboard scale (0=off) |
| `checker_color` | Vec3 | (0.2, 0.2, 0.2) | Secondary checkerboard color |

## Material Presets

| Preset | Function | Description |
|--------|----------|-------------|
| Red Plastic | `red_plastic()` | Classic red diffuse |
| Blue Plastic | `blue_plastic()` | Blue diffuse |
| Mirror | `mirror()` | Perfect mirror (95% reflectivity) |
| Glass | `glass(ior=1.5)` | Transparent glass |
| Water | `water()` | Water (IOR=1.33) |
| Gold | `gold()` | Gold metallic |
| Checkerboard | `checkerboard(scale, light, dark)` | Checkerboard pattern |

## Built-in Scenes

| Scene | Description | Key Features |
|-------|-------------|--------------|
| `demo` | Various primitives on checker ground | Sphere, box, torus, capsule, smooth union, cylinder |
| `chess` | Chess board with king and pawn | Fog, reflectivity, smooth unions |
| `terrain` | Procedural terrain with animated water | Animated water, fog |
| `abstract` | Morphing smooth-union blob animation | Animation, smooth blending |
| `glass` | Refractive glass, SSS, water sphere | Refraction, SSS, glass |
| `fractal` | Mandelbulb with animated power | Fractal, fog |
| `studio` | Three-point lighting showcase | Gold, glass, emissive, matte |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test classes
pytest tests/ -v -k TestVec3
pytest tests/ -v -k TestSDFPrimitives
pytest tests/ -v -k TestRefraction

# Run with coverage
pytest tests/ -v --cov=raymarcher_k9f2
```

The test suite includes 109+ tests covering:
- Vec3 arithmetic, normalization, reflection, refraction, lerp, clamp, pow
- All SDF primitives (surface, inside, outside, distance accuracy)
- CSG operations (union, smooth union, subtraction, intersection)
- Procedural patterns (checkerboard, stripes, gradient)
- Material properties, serialization, presets
- Scene construction, map, map_distance
- Ray marching (hit detection, normal computation, shadow, AO)
- Camera ray generation including DOF
- All 7 scene builders
- Full rendering pipeline for all scenes
- Refraction/reflection logic, IOR correctness
- Config file loading and scene building
- Performance smoke tests

## Performance

This is a pure-Python ray marcher — it prioritizes correctness and readability over speed.
Typical rendering times (single-threaded):

| Resolution | Scene | Time |
|-----------|-------|------|
| 320×240 | demo | ~5s |
| 800×600 | demo | ~45s |
| 800×600 | glass | ~120s |
| 800×600 | fractal | ~180s |

Tips for faster rendering:
- Use `--no-shadow` to disable shadows (2-3× speedup)
- Reduce `--steps` from 128 to 64 for simple scenes
- Use `--samples 1` (default) — increase only for anti-aliasing
- Render at lower resolution and upscale

## Roadmap

- [ ] Numpy-vectorized rendering for 10-50× speedup
- [ ] Environment map / HDRI sky support
- [ ] More primitives (ellipsoid, rounded box, link, etc.)
- [ ] Displacement mapping
- [ ] Volume rendering (clouds, smoke)
- [ ] Multi-threaded rendering via multiprocessing
- [ ] Interactive preview mode (progressive rendering)
- [ ] Export to OBJ/STL for 3D printing
- [ ] Scene graph with transforms (translate, rotate, scale)
- [ ] Tone mapping operators (ACES, Reinhard)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding primitives, scenes, and features.

## License

MIT — see [LICENSE](LICENSE) for details.

## Changelog

### v1.1.0 — Comprehensive Improvement
- **Architecture**: Restructured from single file to proper Python package with modular imports
- **CLI**: Added `raymarcher-k9f2` command with `--config`, `--list-scenes`, `--version`, `--gamma`, `--exposure`, `--verbose` options
- **Config**: Added YAML/TOML scene configuration support
- **Materials**: Added presets (`glass()`, `gold()`, `mirror()`, `water()`, `checkerboard()`) and serialization (`to_dict()`/`from_dict()`)
- **Patterns**: Added `stripes()` and `gradient()` procedural patterns
- **CSG**: Added `sdf_smooth_intersection()`
- **Scenes**: Added `studio` scene with three-point lighting
- **Camera**: Added `to_dict()`/`from_dict()` serialization
- **Renderer**: Added `gamma` and `exposure` parameters, `render_to_pil()` method
- **Vec3**: Added `__eq__`, `to_tuple()`, `from_tuple()` methods
- **CLI entry point**: Installable via `pip install -e .`
- **Tests**: Migrated to pytest with 109+ tests in parametrized test classes
- **CI**: Added GitHub Actions workflow for Python 3.11/3.12
- **Docs**: Added CONTRIBUTING.md, LICENSE, comprehensive README with badges, TOC, and examples
- **Examples**: Added `examples/` directory with YAML config and Python scripts
- **pyproject.toml**: Proper package metadata, dependencies, console script entry point

### v1.0.0 — Initial Release (Phase 1-3)
- Core ray marching engine with 9 SDF primitives
- Physically-based shading (Blinn-Phong, Fresnel)
- Reflections, refractions (glass), subsurface scattering
- Soft shadows, ambient occlusion, depth-of-field
- 6 built-in scenes (demo, chess, terrain, abstract, glass, fractal)
- 7 bug fixes from Phase 3 (3 critical, 1 moderate, 3 minor)
- 109 tests (90 original + 19 bug-specific)