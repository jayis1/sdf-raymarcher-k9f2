# SDF Ray Marcher k9f2

A pure-Python signed distance field (SDF) ray marcher that renders 3D scenes with physically-based lighting, reflections, soft shadows, and ambient occlusion — all without any GPU or graphics API.

## How It Works

The renderer uses **ray marching**: for each pixel, a ray is cast from the camera through the scene. At each step, the ray advances by the distance to the closest surface (the signed distance field value). This continues until the ray either hits a surface (distance < epsilon) or escapes to infinity.

### Key Concepts

- **Signed Distance Fields (SDFs)**: Each object is defined by a mathematical function that returns the shortest distance from any point in space to the surface. Negative values mean "inside" the object.
- **Ray Marching**: Rays advance through the scene in adaptive steps, with step size determined by the SDF value — fast in open space, slow near surfaces.
- **Normal Estimation**: Surface normals are computed via central finite differences of the SDF gradient.
- **CSG Operations**: Boolean union, subtraction, and intersection allow combining primitives, plus smooth blending via exponential smooth union.
- **Physically-Based Shading**: Blinn-Phong specular, Lambert diffuse, Schlick Fresnel for reflections.
- **Soft Shadows**: By tracking closest approach distance along shadow rays, we get realistic penumbra.
- **Ambient Occlusion**: Estimated by sampling the SDF along the surface normal to detect nearby geometry.

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

### CSG Operations

- `sdf_union` — Boolean union (A ∪ B)
- `sdf_smooth_union` — Smooth blended union with configurable radius
- `sdf_subtraction` — Boolean subtraction (A − B)
- `sdf_intersection` — Boolean intersection (A ∩ B)

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

# Abstract animation (8 frames)
python3 raymarcher.py --scene abstract --animate 8

# Terrain with no shadows (faster)
python3 raymarcher.py --scene terrain --no-shadow --output terrain.png
```

### All Options

```
  --width W         Image width (default: 800)
  --height H        Image height (default: 600)
  --samples N       Samples per pixel for anti-aliasing (default: 1)
  --output PATH     Output file path (default: output.png)
  --scene NAME      Scene: demo, chess, terrain, abstract (default: demo)
  --animate N       Render N animation frames (default: 1)
  --max-bounces N   Max reflection bounces (default: 3)
  --ao              Enable ambient occlusion (slower)
  --soft-shadow     Enable soft shadows (slower)
  --no-shadow       Disable shadows entirely
  --steps N         Max ray march steps per ray (default: 128)
```

### Built-in Scenes

- **demo** — Various primitives (sphere, box, torus, capsule, smooth union, cylinder) on a reflective checker ground
- **chess** — A chess board with king and pawn pieces, atmospheric fog
- **terrain** — Procedural terrain with water plane and fog
- **abstract** — Morphing smooth-union blob animation

### Programmatic Usage

```python
from raymarcher import Scene, Camera, Renderer, RayMarcher, Material, Vec3, sdf_sphere, sdf_smooth_union

scene = Scene()
scene.add_object(lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0),
                 Material(albedo=Vec3(1, 0, 0), reflectivity=0.5))
scene.add_directional_light(Vec3(0.5, 0.8, 0.3))

marcher = RayMarcher(enable_shadows=True, enable_ao=True)
renderer = Renderer(width=640, height=480, marcher=marcher)
camera = Camera(Vec3(4, 3, 4), Vec3(0, 1, 0), fov=55)

img = renderer.render(scene, camera)
renderer.save(img, 'my_scene.png')
```

## Architecture

```
raymarcher.py
├── Vec3            — 3D vector math (add, sub, mul, dot, cross, reflect, lerp, clamp)
├── Material        — Surface properties (albedo, specular, roughness, reflectivity, emissive, fresnel)
├── SDF Primitives  — sphere, box, torus, cylinder, capsule, plane, cone
├── CSG Operations  — union, smooth_union, subtraction, intersection
├── SceneObject     — SDF function + material pair
├── Scene           — Object collection, lights, sky evaluation
├── RayMarcher      — Core engine (march, normal, AO, shadow, shade, trace)
├── Camera          — Perspective camera with look-at
└── Renderer        — Pixel loop, anti-aliasing, gamma correction, PNG output
```

## Requirements

- Python 3.11+
- NumPy
- Pillow

## License

MIT