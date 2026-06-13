# SDF Ray Marcher k9f2

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-141%20passing-brightgreen.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

A pure-Python signed distance field (SDF) ray marcher with physically-based lighting, reflections, refractions, subsurface scattering, depth-of-field, tone mapping, SDF transforms, CSG operations, and multiprocessing support.

```
        *     .  *       *   .        .    *        .
   .        .     .         .    *         .    .
      .  *    ┌─────────────────────┐    *  .
   .         │    ◆ SDF Raymarcher │         .
     *       │  Pure-Python 3D     │  .         .
       .     │  Rendering Engine   │     *
  .         └─────────────────────┘        .
```

## ✨ Features

- **🟢 Pure Python** — No GPU, C extensions, or compiled dependencies required
- **🔵 SDF Primitives** — Sphere, Box, Torus, Cylinder, Capsule, Cone, Plane, Ellipsoid, Rounded Box, Link, Hex Prism, Mandelbulb
- **🟡 CSG Operations** — Union, Smooth Union, Subtraction, Intersection
- **🔴 SDF Transforms** — Translate, Rotate (X/Y/Z), Scale, Repeat, Elongate
- **⚪ Physically-Based Lighting** — Blinn-Phong with Fresnel reflections
- **🟣 Glass & Refractions** — Snell's law refraction with proper IOR handling and TIR
- **🟠 Subsurface Scattering** — Approximate SSS for skin, wax, jade
- **🟤 Depth of Field** — Configurable aperture and focal distance
- **⚫ Soft Shadows** — Penumbra-based soft shadows
- **🔵 Ambient Occlusion** — Self-occlusion estimation for concave geometry
- **🟢 Tone Mapping** — ACES, Reinhard, Filmic, and clamped operators with exposure control
- **🟡 Multiprocessing** — Parallel rendering across CPU cores
- **🔴 Material Presets** — Glass, Gold, Mirror, Water, Copper, Jade, Chrome, Marble, Emissive
- **⚪ YAML/TOML Config** — Define scenes without Python code
- **🔵 CLI** — Full command-line interface with animation support
- **🟡 8 Built-in Scenes** — Demo, Chess, Terrain, Abstract, Glass, Fractal, Studio, Showcase

## 📑 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
  - [Command Line](#command-line)
  - [Python API](#python-api)
  - [Scene Configuration](#scene-configuration-yamltoml)
- [Architecture](#-architecture)
- [Primitives](#-primitives)
- [Materials](#-materials)
- [Transforms](#transforms)
- [Tone Mapping](#-tone-mapping)
- [Scenes](#-scenes)
- [Performance](#-performance)
- [Contributing](#-contributing)
- [License](#-license)

## 📦 Installation

### From Source

```bash
git clone https://github.com/jayis1/sdf-raymarcher-k9f2.git
cd sdf-raymarcher-k9f2
pip install -e ".[dev]"
```

### Dependencies

- Python 3.11+
- NumPy
- Pillow
- PyYAML (optional, for YAML config)
- toml (optional, for TOML config)

## 🚀 Quick Start

### Command Line

```bash
# Render the demo scene
raymarcher-k9f2 --scene demo --output demo.png

# Render with tone mapping
raymarcher-k9f2 --scene glass --tonemap aces --exposure 1.2 --output glass.png

# Render with shadows and AO
raymarcher-k9f2 --scene chess --ao --soft-shadow --output chess.png

# Multiprocess rendering
raymarcher-k9f2 --scene showcase --multiprocess --output fast.png

# Animation (8 frames)
raymarcher-k9f2 --scene abstract --animate 8 --output frame.png

# Depth of field
raymarcher-k9f2 --scene glass --dof 8 --aperture 0.15 --output dof.png

# List available scenes and tone mappers
raymarcher-k9f2 --list-scenes
raymarcher-k9f2 --list-tonemaps
```

### Python API

```python
from raymarcher_k9f2 import Scene, Camera, Renderer, RayMarcher, Material, Vec3
from raymarcher_k9f2.primitives import sdf_sphere, sdf_plane
from raymarcher_k9f2.tonemap import apply_tonemap

# Create scene
scene = Scene()
scene.add_object(
    lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0),
    Material(albedo=Vec3(0.8, 0.2, 0.1), specular=0.5, roughness=0.3)
)
scene.add_object(
    lambda p: sdf_plane(p, height=0.0),
    Material(albedo=Vec3(0.5, 0.5, 0.5), checker_scale=2.0, checker_color=Vec3(0.2, 0.2, 0.25))
)
scene.add_directional_light(Vec3(0.5, 0.8, 0.3), Vec3(1.0, 0.95, 0.85), 1.2)

# Render
marcher = RayMarcher(enable_shadows=True, enable_ao=True, max_bounces=4)
renderer = Renderer(width=800, height=600, marcher=marcher)
camera = Camera(Vec3(5, 3, 5), Vec3(0, 1, 0), fov=55)

img = renderer.render(scene, camera)

# Apply tone mapping
img_ldr = apply_tonemap(img, method='aces', exposure=1.0)

# Save
renderer.save(img_ldr, "output.png")
```

### Scene Configuration (YAML/TOML)

```yaml
# scene.yaml
scene:
  objects:
    - type: sphere
      center: [0, 1, 0]
      radius: 1.0
      material:
        albedo: [0.8, 0.2, 0.1]
        specular: 0.5
        roughness: 0.3

    - type: ellipsoid
      center: [3, 1, 0]
      radii: [0.8, 1.2, 0.6]
      material:
        albedo: [0.1, 0.7, 0.4]
        specular: 0.6

    - type: rounded_box
      center: [-3, 0.7, 0]
      half_extents: [0.7, 0.7, 0.7]
      radius: 0.15
      material:
        albedo: [0.8, 0.5, 0.15]

    - type: plane
      height: 0.0
      material:
        albedo: [0.5, 0.5, 0.5]
        checker_scale: 2.0
        checker_color: [0.2, 0.2, 0.25]

  lights:
    - type: directional
      direction: [0.5, 0.8, 0.3]
      intensity: 1.2

renderer:
  width: 800
  height: 600

camera:
  position: [5, 3, 5]
  target: [0, 1, 0]
  fov: 55

output: output.png
```

```bash
raymarcher-k9f2 --config scene.yaml --output custom.png
```

## 🏗 Architecture

The rendering pipeline follows this flow:

```
Camera → Ray Generation → Ray Marching → Scene SDF Evaluation → Normal Estimation → Shading → Tone Mapping → Image
```

**Core modules:**

| Module | Purpose |
|--------|---------|
| `vec3.py` | 3D vector math (Vec3 class) |
| `primitives.py` | SDF primitive functions |
| `transforms.py` | SDF spatial transforms |
| `csg.py` | Constructive Solid Geometry |
| `material.py` | Surface properties and presets |
| `scene.py` | Scene container (objects, lights, environment) |
| `marcher.py` | Ray marching engine (trace, shade, shadow, AO) |
| `camera.py` | Perspective camera with DOF |
| `renderer.py` | Image assembly and output |
| `vec_renderer.py` | Multiprocessing renderer |
| `tonemap.py` | HDR tone mapping operators |
| `scenes.py` | 8 built-in scene presets |
| `config.py` | YAML/TOML scene configuration parser |
| `cli.py` | Command-line interface |

See [docs/architecture.md](docs/architecture.md) for a detailed deep-dive.

## 🔷 Primitives

| Primitive | Function | Description |
|-----------|----------|-------------|
| Sphere | `sdf_sphere(p, center, radius)` | Basic sphere |
| Box | `sdf_box(p, center, half_extents)` | Axis-aligned box |
| Torus | `sdf_torus(p, center, major_r, minor_r)` | Torus ring |
| Cylinder | `sdf_cylinder(p, center, radius, half_height)` | Vertical cylinder |
| Capsule | `sdf_capsule(p, a, b, radius)` | Line-segment capsule |
| Cone | `sdf_cone(p, center, half_angle, height)` | Cone with base |
| Plane | `sdf_plane(p, height)` | Infinite horizontal plane |
| Hex Prism | `sdf_hex_prism(p, center, radius, half_height)` | Hexagonal prism |
| Mandelbulb | `sdf_mandelbulb(p, center, scale, power, iterations)` | Fractal mandelbulb |
| **Ellipsoid** | `sdf_ellipsoid(p, center, radii)` | Stretched sphere |
| **Rounded Box** | `sdf_rounded_box(p, center, half_extents, radius)` | Box with rounded edges |
| **Link** | `sdf_link(p, center, le, r1, r2)` | Chain link (torus pair) |

## 🎨 Materials

### Material Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `albedo` | Vec3 | (0.5, 0.5, 0.5) | Base diffuse color |
| `specular` | float | 0.5 | Specular highlight intensity |
| `roughness` | float | 0.5 | Surface roughness (0=mirror, 1=diffuse) |
| `reflectivity` | float | 0.0 | Mirror reflection strength |
| `fresnel` | float | 0.04 | Fresnel reflection coefficient |
| `transparency` | float | 0.0 | Glass-like transparency |
| `ior` | float | 1.0 | Index of refraction |
| `subsurface` | float | 0.0 | Subsurface scattering amount |
| `subsurface_color` | Vec3 | (1, 0.3, 0.2) | SSS transmission color |
| `emissive` | Vec3 | (0, 0, 0) | Self-illumination color |

### Presets

```python
from raymarcher_k9f2.material import glass, gold, mirror, water, copper, jade, chrome, marble, emissive

mat = glass()              # Clear glass (ior=1.5)
mat = gold()               # Metallic gold
mat = mirror()             # Perfect mirror
mat = water()              # Transparent water (ior=1.33)
mat = copper()             # Copper metallic
mat = jade()               # Translucent green jade
mat = chrome()             # Highly reflective chrome
mat = marble()             # White marble with slight SSS
mat = emissive(Vec3(1, 0.8, 0.3), intensity=2.0)  # Glowing warm light
```

## 🔄 Transforms

Spatial transforms let you position, rotate, and scale SDF primitives:

```python
from raymarcher_k9f2.transforms import (
    sdf_translate, sdf_scale, sdf_rotate_y, sdf_rotate_x,
    sdf_rotate_z, sdf_repeat, sdf_elongate
)
from raymarcher_k9f2.primitives import sdf_sphere

# Sphere at (3, 1, 0)
sphere_at_pos = lambda p: sdf_translate(p, Vec3(3, 1, 0), lambda q: sdf_sphere(q, Vec3(0, 0, 0), 1.0))

# Rotated sphere
rotated = lambda p: sdf_rotate_y(p, math.pi / 4, lambda q: sdf_sphere(q, Vec3(0, 1, 0), 1.0))

# Scaled sphere (0.5x)
scaled = lambda p: sdf_scale(p, 0.5, lambda q: sdf_sphere(q, Vec3(0, 0, 0), 1.0))

# Infinite repetition (brick pattern)
repeated = lambda p: sdf_repeat(p, Vec3(3, 1e6, 3), lambda q: sdf_sphere(q, Vec3(0, 0, 0), 0.3))

# Elongated sphere (capsule-like)
elongated = lambda p: sdf_elongate(p, Vec3(0, 1, 0), lambda q: sdf_sphere(q, Vec3(0, 0, 0), 0.5))
```

## 🎭 Tone Mapping

```python
from raymarcher_k9f2.tonemap import apply_tonemap, list_tonemaps

# Available methods
print(list_tonemaps())  # ['linear', 'reinhard', 'aces', 'filmic', 'clamped']

# Apply ACES tone mapping (industry standard)
img_ldr = apply_tonemap(img, method='aces', exposure=1.0)

# Apply Reinhard
img_ldr = apply_tonemap(img, method='reinhard', exposure=1.5)

# From CLI
# raymarcher-k9f2 --scene glass --tonemap aces --exposure 1.2 --output hdr.png
```

| Method | Description | Best For |
|--------|-------------|----------|
| `aces` | Industry standard filmic curve | Most scenes, cinematic look |
| `reinhard` | `x / (1+x)` simple mapping | Soft, natural look |
| `filmic` | Uncharted 2 curve | High contrast, dramatic |
| `clamped` | Hard clamp to [0, 1] | Debug visualization |
| `linear` | No mapping (pass-through) | HDR processing |

## 🎬 Scenes

8 built-in scenes available via CLI:

```bash
raymarcher-k9f2 --scene demo        # Classic sphere-on-plane demo
raymarcher-k9f2 --scene chess        # Metallic chess pieces
raymarcher-k9f2 --scene terrain      # Procedural landscape
raymarcher-k9f2 --scene abstract     # Smooth blob compositions
raymarcher-k9f2 --scene glass        # Refractive glass spheres
raymarcher-k9f2 --scene fractal      # Mandelbulb fractal
raymarcher-k9f2 --scene studio       # Studio lighting setup
raymarcher-k9f2 --scene showcase     # New primitives + tone mapping demo
```

## ⚡ Performance

| Resolution | Scene | Single Core | 8-Core Multiprocess |
|-----------|-------|-------------|---------------------|
| 320×240 | demo | ~5s | ~1s |
| 800×600 | demo | ~45s | ~8s |
| 800×600 | glass | ~120s | ~20s |

Use `--multiprocess` for parallel rendering:

```bash
raymarcher-k9f2 --scene demo --width 800 --height 600 --multiprocess --output fast.png
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run only new feature tests
pytest tests/test_new_features.py -v

# Run with coverage
pytest tests/ --cov=raymarcher_k9f2
```

141 tests covering:
- Vec3 math operations
- All SDF primitives (inside/outside/surface)
- CSG operations
- Material presets
- Camera and ray generation
- Ray marching and shading
- Tone mapping operators
- SDF transforms
- Config parsing
- Full rendering pipeline
- Multiprocessing renderer

## 📁 Project Structure

```
sdf-raymarcher-k9f2/
├── src/raymarcher_k9f2/
│   ├── __init__.py        # Package exports
│   ├── vec3.py            # 3D vector math
│   ├── primitives.py      # SDF primitive functions
│   ├── transforms.py      # SDF spatial transforms
│   ├── csg.py             # CSG operations
│   ├── patterns.py        # Checker/stripe patterns
│   ├── material.py        # Material definitions & presets
│   ├── scene.py           # Scene container
│   ├── camera.py          # Perspective camera
│   ├── marcher.py         # Ray marching engine
│   ├── renderer.py        # Image assembly
│   ├── vec_renderer.py    # Multiprocessing renderer
│   ├── tonemap.py         # Tone mapping operators
│   ├── scenes.py          # Built-in scene presets
│   ├── config.py          # YAML/TOML config parser
│   └── cli.py             # CLI entry point
├── tests/
│   ├── test_raymarcher.py  # Core test suite
│   └── test_new_features.py # New feature tests
├── examples/
│   ├── simple_sphere.py    # Basic sphere rendering
│   ├── glass_scene.py      # Glass refraction demo
│   ├── transforms_example.py # Transform demo
│   ├── tonemap_example.py   # Tone mapping demo
│   └── scene.yaml          # YAML config example
├── docs/
│   └── architecture.md     # Architecture deep-dive
├── pyproject.toml          # Package configuration
├── README.md               # This file
├── CONTRIBUTING.md         # Contribution guide
└── LICENSE                 # MIT License
```

## 🗺 Roadmap

- [ ] GPU acceleration via CUDA/Numba
- [ ] BVH-based acceleration structure for many-object scenes
- [ ] Image-based lighting (IBL) with HDR environment maps
- [ ] Progressive rendering with interactive preview
- [ ] Animation keyframing system
- [ ] Volume rendering (clouds, fog, smoke)
- [ ] Distributed rendering across multiple machines
- [ ] Web-based viewer with WebSocket streaming
- [ ] More SDF primitives (octahedron, dodecahedron, gyroid)
- [ ] Normal map / depth map / AO map output passes

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Bug reports, feature requests, and pull requests are welcome!

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Credits

Built with [SDF Ray Marcher k9f2](https://github.com/jayis1/sdf-raymarcher-k9f2) — a pure-Python 3D rendering engine.

### Recent Improvements (v2.0.0)

- **Tone Mapping**: ACES, Reinhard, Filmic, and clamped operators with exposure control
- **New Primitives**: Ellipsoid, Rounded Box, Link (chain link)
- **SDF Transforms**: Translate, Rotate (X/Y/Z), Scale, Repeat, Elongate
- **Multiprocessing Renderer**: Parallel rendering across CPU cores
- **Material Presets**: Copper, Jade, Chrome, Marble, Emissive
- **8th Scene**: Showcase scene with new primitives and cinematic lighting
- **YAML/TOML Config**: Support for new primitives in scene files
- **Architecture Documentation**: Detailed pipeline and module guide
- **Comprehensive Tests**: 141 tests covering all features
- **CLI Enhancements**: `--tonemap`, `--exposure`, `--list-tonemaps`, `--multiprocess`