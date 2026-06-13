#!/usr/bin/env python3
"""
SDF Ray Marcher — A pure-Python signed distance field ray marcher.

Renders 3D scenes using ray marching against analytic signed distance fields.
Supports multiple primitives, materials, soft shadows, ambient occlusion,
reflections, and animation.

Usage:
    source venv/bin/activate
    python3 raymarcher.py [options]

Options:
    --width W        Image width (default: 800)
    --height H       Image height (default: 600)
    --samples N      Samples per pixel for anti-aliasing (default: 1)
    --output PATH    Output file path (default: output.png)
    --scene NAME     Scene to render: demo, chess, terrain, abstract (default: demo)
    --animate N       Render N animation frames (default: 1, single frame)
    --max-bounces N  Max reflection bounces (default: 3)
    --ao             Enable ambient occlusion (slower)
    --soft-shadow    Enable soft shadows (slower)
    --no-shadow      Disable shadows entirely
"""

import math
import argparse
import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple

try:
    import numpy as np
    from PIL import Image
except ImportError:
    print("Missing dependencies. Run: source venv/bin/activate && pip install numpy Pillow")
    raise SystemExit(1)


# ─── Vector Math ────────────────────────────────────────────────────────────

class Vec3:
    """Minimal 3D vector class for ray marching math."""
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other: 'Vec3') -> 'Vec3':
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vec3') -> 'Vec3':
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other) -> 'Vec3':
        if isinstance(other, Vec3):
            return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        return Vec3(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, scalar: float) -> 'Vec3':
        return self.__mul__(scalar)

    def __neg__(self) -> 'Vec3':
        return Vec3(-self.x, -self.y, -self.z)

    def __repr__(self) -> str:
        return f"Vec3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    def dot(self, other: 'Vec3') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vec3') -> 'Vec3':
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> 'Vec3':
        l = self.length()
        if l < 1e-12:
            return Vec3(0, 0, 0)
        inv = 1.0 / l
        return Vec3(self.x * inv, self.y * inv, self.z * inv)

    def reflect(self, normal: 'Vec3') -> 'Vec3':
        """Reflect this vector off a surface with given normal."""
        return self - normal * (2.0 * self.dot(normal))

    def lerp(self, other: 'Vec3', t: float) -> 'Vec3':
        return self * (1.0 - t) + other * t

    def clamp(self, lo: float = 0.0, hi: float = 1.0) -> 'Vec3':
        return Vec3(
            max(lo, min(hi, self.x)),
            max(lo, min(hi, self.y)),
            max(lo, min(hi, self.z)),
        )


# ─── Color Constants ────────────────────────────────────────────────────────

BLACK = Vec3(0, 0, 0)
WHITE = Vec3(1, 1, 1)
SKY_BLUE = Vec3(0.5, 0.7, 1.0)


# ─── Material ────────────────────────────────────────────────────────────────

@dataclass
class Material:
    """Surface material properties."""
    albedo: Vec3 = field(default_factory=lambda: Vec3(0.8, 0.8, 0.8))
    specular: float = 0.5       # Specular highlight strength
    roughness: float = 0.3      # 0=mirror, 1=diffuse
    reflectivity: float = 0.0    # 0=no reflection, 1=perfect mirror
    emissive: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    fresnel: float = 0.04        # Schlick fresnel term


# ─── SDF Primitives ─────────────────────────────────────────────────────────

def sdf_sphere(p: Vec3, center: Vec3, radius: float) -> float:
    return (p - center).length() - radius


def sdf_box(p: Vec3, center: Vec3, half_extents: Vec3) -> float:
    q = p - center
    # Component-wise absolute and clamping
    qx, qy, qz = abs(q.x), abs(q.y), abs(q.z)
    cx = max(qx - half_extents.x, 0.0)
    cy = max(qy - half_extents.y, 0.0)
    cz = max(qz - half_extents.z, 0.0)
    outside_dist = math.sqrt(cx * cx + cy * cy + cz * cz)
    inside_dist = max(half_extents.x - qx, half_extents.y - qy, half_extents.z - qz)
    return outside_dist if inside_dist < 0 else -inside_dist


def sdf_torus(p: Vec3, center: Vec3, major_r: float, minor_r: float) -> float:
    q = p - center
    xz_dist = math.sqrt(q.x * q.x + q.z * q.z) - major_r
    return math.sqrt(xz_dist * xz_dist + q.y * q.y) - minor_r


def sdf_cylinder(p: Vec3, center: Vec3, radius: float, half_height: float) -> float:
    q = p - center
    dist_xz = math.sqrt(q.x * q.x + q.z * q.z) - radius
    dist_y = abs(q.y) - half_height
    outside = math.sqrt(max(dist_xz, 0.0) ** 2 + max(dist_y, 0.0) ** 2)
    inside = min(max(dist_xz, dist_y), 0.0)
    return outside + inside


def sdf_capsule(p: Vec3, a: Vec3, b: Vec3, radius: float) -> float:
    ab = b - a
    ap = p - a
    t = ap.dot(ab) / ab.dot(ab)
    t = max(0.0, min(1.0, t))
    closest = a + ab * t
    return (p - closest).length() - radius


def sdf_plane(p: Vec3, height: float = 0.0) -> float:
    return p.y - height


def sdf_cone(p: Vec3, center: Vec3, half_angle: float, height: float) -> float:
    """Cone pointing up from center.y, given half_angle in radians and height."""
    q = p - center
    q.y -= height
    # Cone SDF
    sin_a = math.sin(half_angle)
    cos_a = math.cos(half_angle)
    # Projected distance along cone surface
    d = math.sqrt(q.x * q.x + q.z * q.z)
    inside = -q.y - d * cos_a
    outside = d * sin_a + q.y * cos_a
    if inside < 0 and outside < 0:
        return -min(-inside, -outside)
    return max(outside, 0.0)


# ─── CSG Operations ──────────────────────────────────────────────────────────

def sdf_union(d1: float, d2: float) -> float:
    return min(d1, d2)


def sdf_smooth_union(d1: float, d2: float, k: float = 0.5) -> float:
    """Smooth boolean union with blending factor k."""
    h = max(k - abs(d1 - d2), 0.0) / k
    return min(d1, d2) - h * h * k * 0.25


def sdf_subtraction(d1: float, d2: float) -> float:
    """Subtract d2 from d1 (d1 - d2)."""
    return max(-d2, d1)


def sdf_intersection(d1: float, d2: float) -> float:
    return max(d1, d2)


# ─── Scene Objects ───────────────────────────────────────────────────────────

@dataclass
class SceneObject:
    """An object in the scene: SDF function + material."""
    sdf_func: Callable[[Vec3], float]
    material: Material


# ─── Hit Info ────────────────────────────────────────────────────────────────

@dataclass
class HitResult:
    distance: float = float('inf')
    material: Optional[Material] = None
    object_index: int = -1


# ─── Lighting ────────────────────────────────────────────────────────────────

@dataclass
class PointLight:
    position: Vec3
    color: Vec3 = field(default_factory=lambda: Vec3(1, 1, 1))
    intensity: float = 1.0


@dataclass
class DirectionalLight:
    direction: Vec3  # Points toward the light source
    color: Vec3 = field(default_factory=lambda: Vec3(1, 1, 1))
    intensity: float = 1.0


# ─── Scene ───────────────────────────────────────────────────────────────────

class Scene:
    """Container for scene objects, lights, and the SDF evaluation function."""

    def __init__(self):
        self.objects: List[SceneObject] = []
        self.directional_lights: List[DirectionalLight] = []
        self.point_lights: List[PointLight] = []
        self.sky_color: Vec3 = SKY_BLUE
        self.fog_color: Vec3 = Vec3(0.7, 0.75, 0.85)
        self.fog_density: float = 0.0
        self.time: float = 0.0
        self.background_func: Optional[Callable[[Vec3], Vec3]] = None

    def add_object(self, sdf_func: Callable[[Vec3], float], material: Material) -> int:
        idx = len(self.objects)
        self.objects.append(SceneObject(sdf_func, material))
        return idx

    def add_directional_light(self, direction: Vec3, color: Vec3 = None, intensity: float = 1.0):
        if color is None:
            color = Vec3(1, 1, 1)
        self.directional_lights.append(DirectionalLight(direction.normalized(), color, intensity))

    def add_point_light(self, position: Vec3, color: Vec3 = None, intensity: float = 1.0):
        if color is None:
            color = Vec3(1, 1, 1)
        self.point_lights.append(PointLight(position, color, intensity))

    def map(self, p: Vec3) -> HitResult:
        """Evaluate the SDF at point p, returning the closest object info."""
        closest_dist = float('inf')
        closest_idx = -1
        for i, obj in enumerate(self.objects):
            d = obj.sdf_func(p)
            if d < closest_dist:
                closest_dist = d
                closest_idx = i
        hit = HitResult(distance=closest_dist, object_index=closest_idx)
        if closest_idx >= 0:
            hit.material = self.objects[closest_idx].material
        return hit

    def map_distance(self, p: Vec3) -> float:
        """Fast path: just the distance value, no material lookup."""
        min_d = float('inf')
        for obj in self.objects:
            d = obj.sdf_func(p)
            if d < min_d:
                min_d = d
        return min_d

    def get_sky(self, direction: Vec3) -> Vec3:
        """Sample sky color for a given direction."""
        if self.background_func:
            return self.background_func(direction)
        # Default gradient sky
        t = max(direction.y, 0.0)
        horizon = Vec3(0.8, 0.82, 0.85)
        zenith = Vec3(0.2, 0.35, 0.75)
        sky = horizon.lerp(zenith, t)
        # Sun glow
        sun_dir = Vec3(0.5, 0.8, 0.3).normalized()
        sun_dot = max(direction.dot(sun_dir), 0.0)
        sun = Vec3(1.0, 0.95, 0.8) * (sun_dot ** 128) * 3.0
        glow = Vec3(1.0, 0.9, 0.7) * (sun_dot ** 8) * 0.3
        return sky + sun + glow


# ─── Ray Marcher ────────────────────────────────────────────────────────────

class RayMarcher:
    """Core ray marching engine with configurable quality settings."""

    def __init__(self, max_steps: int = 128, max_distance: float = 200.0,
                 surface_epsilon: float = 0.001, normal_epsilon: float = 0.002,
                 max_bounces: int = 3, enable_shadows: bool = True,
                 enable_soft_shadows: bool = False, shadow_softness: float = 8.0,
                 enable_ao: bool = False, ao_steps: int = 5, ao_step_size: float = 0.1,
                 ao_strength: float = 1.0):
        self.max_steps = max_steps
        self.max_distance = max_distance
        self.surface_epsilon = surface_epsilon
        self.normal_epsilon = normal_epsilon
        self.max_bounces = max_bounces
        self.enable_shadows = enable_shadows
        self.enable_soft_shadows = enable_soft_shadows
        self.shadow_softness = shadow_softness
        self.enable_ao = enable_ao
        self.ao_steps = ao_steps
        self.ao_step_size = ao_step_size
        self.ao_strength = ao_strength

    def march(self, origin: Vec3, direction: Vec3, scene: Scene) -> HitResult:
        """March a ray through the scene, returning the closest hit."""
        t = 0.0
        hit = HitResult()
        min_dist = float('inf')

        for step in range(self.max_steps):
            p = origin + direction * t
            result = scene.map(p)
            d = result.distance

            # Track the closest we ever got (for soft shadows later)
            if d < min_dist:
                min_dist = d

            if d < self.surface_epsilon:
                hit.distance = t
                hit.material = result.material
                hit.object_index = result.object_index
                return hit

            t += d

            if t > self.max_distance:
                break

        hit.distance = min_dist  # No exact hit, use closest approach
        return hit

    def compute_normal(self, p: Vec3, scene: Scene) -> Vec3:
        """Compute surface normal via central differences of the SDF."""
        e = self.normal_epsilon
        dx = scene.map_distance(Vec3(p.x + e, p.y, p.z)) - scene.map_distance(Vec3(p.x - e, p.y, p.z))
        dy = scene.map_distance(Vec3(p.x, p.y + e, p.z)) - scene.map_distance(Vec3(p.x, p.y - e, p.z))
        dz = scene.map_distance(Vec3(p.x, p.y, p.z + e)) - scene.map_distance(Vec3(p.x, p.y, p.z - e))
        return Vec3(dx, dy, dz).normalized()

    def compute_ao(self, p: Vec3, normal: Vec3, scene: Scene) -> float:
        """Compute ambient occlusion by sampling the SDF along the normal."""
        ao = 0.0
        scale = 1.0
        for i in range(self.ao_steps):
            dist = self.ao_step_size * (i + 1)
            sample_point = p + normal * dist
            d = scene.map_distance(sample_point)
            ao += (dist - d) * scale
            scale *= 0.5
        return max(0.0, 1.0 - self.ao_strength * ao)

    def compute_shadow(self, p: Vec3, light_dir: Vec3, scene: Scene) -> float:
        """Compute hard or soft shadow factor for point p toward light_dir."""
        if not self.enable_shadows:
            return 1.0

        origin = p + light_dir * self.surface_epsilon * 2.0
        t = self.surface_epsilon * 4.0
        shadow = 1.0

        for step in range(64):
            q = origin + light_dir * t
            d = scene.map_distance(q)

            if self.enable_soft_shadows:
                # Soft shadow: penalize close approaches
                penumbra = max(d / t, 0.0)
                shadow = min(shadow, self.shadow_softness * penumbra)
            else:
                if d < self.surface_epsilon:
                    return 0.0  # Hard shadow: fully occluded

            t += d
            if t > self.max_distance:
                break

        return max(0.0, min(1.0, shadow))

    def shade(self, p: Vec3, normal: Vec3, view_dir: Vec3,
              material: Material, scene: Scene) -> Vec3:
        """Shade a surface point with all lighting contributions."""
        color = material.albedo

        # Emissive
        if material.emissive.length() > 0:
            color = material.emissive
            return color

        result = BLACK

        # Ambient
        ambient_strength = 0.15
        if self.enable_ao:
            ambient_strength *= self.compute_ao(p, normal, scene)
        result = result + color * ambient_strength

        # Directional lights
        for light in scene.directional_lights:
            light_dir = light.direction
            n_dot_l = max(normal.dot(light_dir), 0.0)

            # Shadow
            shadow = self.compute_shadow(p, light_dir, scene)

            # Diffuse
            diffuse = color * n_dot_l * light.color * light.intensity * shadow

            # Specular (Blinn-Phong)
            half_vec = (light_dir + view_dir).normalized()
            n_dot_h = max(normal.dot(half_vec), 0.0)
            spec_pow = max(1.0, (1.0 - material.roughness) * 256.0)
            specular = Vec3(1, 1, 1) * (n_dot_h ** spec_pow) * material.specular * shadow

            result = result + diffuse + specular

        # Point lights
        for light in scene.point_lights:
            to_light = light.position - p
            dist = to_light.length()
            light_dir = to_light * (1.0 / dist)
            attenuation = 1.0 / (1.0 + 0.09 * dist + 0.032 * dist * dist)
            n_dot_l = max(normal.dot(light_dir), 0.0)

            shadow = self.compute_shadow(p, light_dir, scene)

            diffuse = color * n_dot_l * light.color * light.intensity * attenuation * shadow

            half_vec = (light_dir + view_dir).normalized()
            n_dot_h = max(normal.dot(half_vec), 0.0)
            spec_pow = max(1.0, (1.0 - material.roughness) * 256.0)
            specular = Vec3(1, 1, 1) * (n_dot_h ** spec_pow) * material.specular * shadow

            result = result + diffuse + specular

        return result.clamp(0.0, 10.0)

    def trace(self, origin: Vec3, direction: Vec3, scene: Scene, bounce: int = 0) -> Vec3:
        """Trace a single ray, potentially bouncing for reflections."""
        hit = self.march(origin, direction, scene)

        if hit.material is None or hit.distance >= self.max_distance * 0.99:
            return scene.get_sky(direction)

        p = origin + direction * hit.distance
        normal = self.compute_normal(p, scene)
        view_dir = (-direction).normalized()

        # Make sure normal faces the ray
        if normal.dot(direction) > 0:
            normal = -normal

        color = self.shade(p, normal, view_dir, hit.material, scene)

        # Reflections
        if hit.material.reflectivity > 0 and bounce < self.max_bounces:
            reflect_dir = direction.reflect(normal)
            reflect_origin = p + normal * self.surface_epsilon * 2.0
            reflect_color = self.trace(reflect_origin, reflect_dir, scene, bounce + 1)
            # Fresnel approximation (Schlick)
            cos_theta = max(view_dir.dot(normal), 0.0)
            fresnel = hit.material.fresnel + (1.0 - hit.material.fresnel) * ((1.0 - cos_theta) ** 5)
            reflect_amount = max(hit.material.reflectivity, fresnel) * (1.0 - hit.material.roughness)
            color = color * (1.0 - reflect_amount) + reflect_color * reflect_amount

        # Fog
        if scene.fog_density > 0:
            fog_factor = 1.0 - math.exp(-scene.fog_density * hit.distance)
            color = color.lerp(scene.fog_color, fog_factor)

        return color.clamp(0.0, 1.0)


# ─── Camera ──────────────────────────────────────────────────────────────────

class Camera:
    """Perspective camera with look-at targeting."""

    def __init__(self, position: Vec3, target: Vec3, fov: float = 60.0, up: Vec3 = None):
        if up is None:
            up = Vec3(0, 1, 0)
        self.position = position
        self.target = target
        self.fov = fov
        self.up = up

    def get_ray(self, u: float, v: float) -> Tuple[Vec3, Vec3]:
        """Get ray direction for normalized screen coordinates u,v in [-0.5, 0.5]."""
        forward = (self.target - self.position).normalized()
        right = forward.cross(self.up).normalized()
        true_up = right.cross(forward).normalized()

        aspect = 1.0  # Will be adjusted in renderer
        fov_rad = math.radians(self.fov)
        half_h = math.tan(fov_rad / 2.0)
        half_w = half_h * aspect

        direction = (forward + right * (u * 2.0 * half_w) + true_up * (v * 2.0 * half_h)).normalized()
        return self.position, direction


# ─── Renderer ────────────────────────────────────────────────────────────────

class Renderer:
    """Turns scenes into images using the ray marcher."""

    def __init__(self, width: int = 800, height: int = 600, samples: int = 1,
                 marcher: RayMarcher = None):
        self.width = width
        self.height = height
        self.samples = samples
        self.marcher = marcher or RayMarcher()

    def render(self, scene: Scene, camera: Camera) -> np.ndarray:
        """Render the scene, returning an (H, W, 3) float array."""
        img = np.zeros((self.height, self.width, 3), dtype=np.float64)
        aspect = self.width / self.height
        fov_rad = math.radians(camera.fov)
        half_h = math.tan(fov_rad / 2.0)
        half_w = half_h * aspect

        forward = (camera.target - camera.position).normalized()
        right = forward.cross(camera.up).normalized()
        true_up = right.cross(forward).normalized()

        total_pixels = self.width * self.height
        last_pct = -1

        t_start = time.time()

        for y in range(self.height):
            pct = int((y * self.width) / total_pixels * 100)
            if pct != last_pct and pct % 10 == 0:
                elapsed = time.time() - t_start
                last_pct = pct
                print(f"  Rendering: {pct}% ({elapsed:.1f}s)")

            for x in range(self.width):
                color = BLACK
                for s in range(self.samples):
                    # Jittered sampling
                    if self.samples > 1:
                        jx = (x + (s * 0.618033988749895 % 1.0)) / self.width
                        jy = (y + (s * 0.414213562373095 % 1.0)) / self.height
                    else:
                        jx = (x + 0.5) / self.width
                        jy = (y + 0.5) / self.height

                    u = (jx - 0.5) * 2.0 * half_w
                    v = (0.5 - jy) * 2.0 * half_h

                    direction = (forward + right * u + true_up * v).normalized()
                    sample_color = self.marcher.trace(camera.position, direction, scene)
                    color = color + sample_color

                color = color * (1.0 / self.samples)
                # Gamma correction
                color = Vec3(
                    pow(max(color.x, 0.0), 1.0 / 2.2),
                    pow(max(color.y, 0.0), 1.0 / 2.2),
                    pow(max(color.z, 0.0), 1.0 / 2.2),
                )
                img[y, x, 0] = color.x
                img[y, x, 1] = color.y
                img[y, x, 2] = color.z

        elapsed = time.time() - t_start
        print(f"  Rendering: 100% ({elapsed:.1f}s)")
        return img

    def save(self, img: np.ndarray, path: str):
        """Save float image array as 8-bit PNG."""
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, 'RGB')
        pil_img.save(path)
        print(f"  Saved: {path} ({img.shape[1]}x{img.shape[0]})")


# ─── Scene Builders ──────────────────────────────────────────────────────────

def build_demo_scene(time: float = 0.0) -> Scene:
    """Default demo scene with various primitives."""
    scene = Scene()
    scene.time = time

    # Ground plane - checkerboard material
    ground_mat = Material(
        albedo=Vec3(0.45, 0.45, 0.45),
        specular=0.2,
        roughness=0.7,
        reflectivity=0.15,
    )

    def ground_sdf(p: Vec3) -> float:
        return sdf_plane(p, height=0.0)

    scene.add_object(ground_sdf, ground_mat)

    # Animated sphere
    def sphere_sdf(p: Vec3) -> float:
        center = Vec3(0, 1.0 + 0.3 * math.sin(time * 1.5), 0)
        return sdf_sphere(p, center, 1.0)

    sphere_mat = Material(
        albedo=Vec3(0.9, 0.2, 0.15),
        specular=0.9,
        roughness=0.05,
        reflectivity=0.4,
        fresnel=0.05,
    )
    scene.add_object(sphere_sdf, sphere_mat)

    # Box
    def box_sdf(p: Vec3) -> float:
        center = Vec3(3.0, 0.75, 1.0)
        return sdf_box(p, center, Vec3(0.75, 0.75, 0.75))

    box_mat = Material(
        albedo=Vec3(0.2, 0.5, 0.9),
        specular=0.6,
        roughness=0.2,
        reflectivity=0.1,
    )
    scene.add_object(box_sdf, box_mat)

    # Torus
    def torus_sdf(p: Vec3) -> float:
        center = Vec3(-2.5, 0.8, 2.0)
        return sdf_torus(p, center, 0.8, 0.25)

    torus_mat = Material(
        albedo=Vec3(0.9, 0.85, 0.1),
        specular=0.7,
        roughness=0.15,
    )
    scene.add_object(torus_sdf, torus_mat)

    # Capsule
    def capsule_sdf(p: Vec3) -> float:
        return sdf_capsule(p, Vec3(-3.0, 0.5, -1.5), Vec3(-3.0, 2.5, -1.5), 0.35)

    capsule_mat = Material(
        albedo=Vec3(0.1, 0.8, 0.4),
        specular=0.5,
        roughness=0.3,
    )
    scene.add_object(capsule_sdf, capsule_mat)

    # Smooth union of two spheres
    def smooth_sdf(p: Vec3) -> float:
        d1 = sdf_sphere(p, Vec3(1.5, 0.7, -2.5), 0.7)
        d2 = sdf_sphere(p, Vec3(2.5, 0.7, -2.5), 0.7)
        return sdf_smooth_union(d1, d2, k=0.5)

    smooth_mat = Material(
        albedo=Vec3(0.85, 0.55, 0.15),
        specular=0.4,
        roughness=0.4,
    )
    scene.add_object(smooth_sdf, smooth_mat)

    # Cylinder
    def cylinder_sdf(p: Vec3) -> float:
        return sdf_cylinder(p, Vec3(3.5, 0.6, -3.0), 0.5, 0.6)

    cyl_mat = Material(
        albedo=Vec3(0.7, 0.3, 0.8),
        specular=0.5,
        roughness=0.2,
    )
    scene.add_object(cylinder_sdf, cyl_mat)

    # Lights
    scene.add_directional_light(Vec3(0.5, 0.8, 0.3), Vec3(1.0, 0.95, 0.85), 1.0)
    scene.add_directional_light(Vec3(-0.3, 0.4, -0.5), Vec3(0.3, 0.4, 0.6), 0.3)
    scene.add_point_light(Vec3(2, 3, -2), Vec3(1.0, 0.8, 0.5), 5.0)

    return scene


def build_chess_scene(time: float = 0.0) -> Scene:
    """Chess-like scene with a reflective board and pieces."""
    scene = Scene()
    scene.time = time
    scene.fog_density = 0.015
    scene.fog_color = Vec3(0.35, 0.3, 0.25)

    # Board
    board_mat = Material(albedo=Vec3(0.15, 0.12, 0.1), specular=0.3, roughness=0.6, reflectivity=0.3)

    def board_sdf(p: Vec3) -> float:
        return sdf_box(p, Vec3(0, -0.1, 0), Vec3(5, 0.1, 5))

    scene.add_object(board_sdf, board_mat)

    # Checker pattern via separate raised squares
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

    # King piece (tall cylinder + sphere)
    king_mat = Material(albedo=Vec3(0.95, 0.92, 0.85), specular=0.8, roughness=0.1, reflectivity=0.2)

    def king_sdf(p: Vec3) -> float:
        body = sdf_cylinder(p, Vec3(0, 0.7, 0), 0.25, 0.7)
        head = sdf_sphere(p, Vec3(0, 1.6, 0), 0.22)
        base = sdf_cylinder(p, Vec3(0, 0.15, 0), 0.35, 0.15)
        return sdf_smooth_union(sdf_smooth_union(body, head, k=0.15), base, k=0.1)

    scene.add_object(king_sdf, king_mat)

    # Pawn (short)
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


def build_terrain_scene(time: float = 0.0) -> Scene:
    """Procedural terrain scene using noise-like SDF."""
    scene = Scene()
    scene.time = time
    scene.fog_density = 0.02
    scene.fog_color = Vec3(0.6, 0.7, 0.8)

    def noise2d(x: float, z: float) -> float:
        """Cheap pseudo-noise for terrain height."""
        return (math.sin(x * 0.5) * math.cos(z * 0.7) * 0.5 +
                math.sin(x * 1.3 + z * 0.9) * 0.25 +
                math.sin(x * 2.7 + z * 2.1) * 0.125)

    def terrain_sdf(p: Vec3) -> float:
        h = noise2d(p.x, p.z) + 1.0
        return p.y - h

    terrain_mat = Material(
        albedo=Vec3(0.35, 0.55, 0.25),
        specular=0.1,
        roughness=0.9,
    )
    scene.add_object(terrain_sdf, terrain_mat)

    # Water plane
    def water_sdf(p: Vec3) -> float:
        return sdf_plane(p, height=0.3)

    water_mat = Material(
        albedo=Vec3(0.15, 0.3, 0.6),
        specular=0.9,
        roughness=0.05,
        reflectivity=0.6,
    )
    scene.add_object(water_sdf, water_mat)

    scene.add_directional_light(Vec3(0.3, 0.8, 0.5), Vec3(1.0, 0.9, 0.7), 1.0)
    scene.add_directional_light(Vec3(-0.5, 0.3, -0.3), Vec3(0.4, 0.5, 0.7), 0.3)

    return scene


def build_abstract_scene(time: float = 0.0) -> Scene:
    """Abstract scene with morphing smooth unions."""
    scene = Scene()
    scene.time = time

    def abstract_sdf(p: Vec3) -> float:
        d = float('inf')
        for i in range(5):
            angle = time * 0.5 + i * math.pi * 2.0 / 5.0
            cx = 2.0 * math.cos(angle)
            cy = 1.5 + 0.5 * math.sin(time + i)
            cz = 2.0 * math.sin(angle)
            d = sdf_smooth_union(d, sdf_sphere(p, Vec3(cx, cy, cz), 0.8), k=1.2)
        return d

    abstract_mat = Material(
        albedo=Vec3(0.8, 0.3, 0.7),
        specular=0.8,
        roughness=0.1,
        reflectivity=0.5,
        fresnel=0.03,
    )
    scene.add_object(abstract_sdf, abstract_mat)

    # Ground
    ground_mat = Material(albedo=Vec3(0.2, 0.2, 0.25), specular=0.2, roughness=0.8, reflectivity=0.3)
    scene.add_object(lambda p: sdf_plane(p, height=0.0), ground_mat)

    scene.add_directional_light(Vec3(0.5, 0.8, 0.3), Vec3(1.0, 0.95, 0.85), 1.0)
    scene.add_directional_light(Vec3(-0.3, 0.5, -0.7), Vec3(0.5, 0.3, 0.7), 0.4)

    return scene


SCENES = {
    'demo': build_demo_scene,
    'chess': build_chess_scene,
    'terrain': build_terrain_scene,
    'abstract': build_abstract_scene,
}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='SDF Ray Marcher')
    parser.add_argument('--width', type=int, default=800, help='Image width')
    parser.add_argument('--height', type=int, default=600, help='Image height')
    parser.add_argument('--samples', type=int, default=1, help='Samples per pixel (AA)')
    parser.add_argument('--output', type=str, default='output.png', help='Output file path')
    parser.add_argument('--scene', type=str, default='demo', choices=SCENES.keys(), help='Scene to render')
    parser.add_argument('--animate', type=int, default=1, help='Number of animation frames')
    parser.add_argument('--max-bounces', type=int, default=3, help='Max reflection bounces')
    parser.add_argument('--ao', action='store_true', help='Enable ambient occlusion')
    parser.add_argument('--soft-shadow', action='store_true', help='Enable soft shadows')
    parser.add_argument('--no-shadow', action='store_true', help='Disable shadows')
    parser.add_argument('--steps', type=int, default=128, help='Max ray march steps')

    args = parser.parse_args()

    marcher = RayMarcher(
        max_steps=args.steps,
        max_bounces=args.max_bounces,
        enable_shadows=not args.no_shadow,
        enable_soft_shadows=args.soft_shadow,
        enable_ao=args.ao,
    )

    renderer = Renderer(
        width=args.width,
        height=args.height,
        samples=args.samples,
        marcher=marcher,
    )

    build_scene = SCENES[args.scene]

    if args.animate > 1:
        # Animation mode
        import os
        basename = os.path.splitext(args.output)[0]
        ext = os.path.splitext(args.output)[1] or '.png'
        for frame in range(args.animate):
            t = frame / args.animate * math.pi * 2.0
            print(f"Frame {frame + 1}/{args.animate}")
            scene = build_scene(t)
            if args.scene == 'demo':
                cam = Camera(Vec3(7 * math.cos(t * 0.3), 4, 7 * math.sin(t * 0.3)),
                             Vec3(0, 1, 0), fov=55)
            elif args.scene == 'abstract':
                cam = Camera(Vec3(6 * math.cos(t * 0.2), 4, 6 * math.sin(t * 0.2)),
                             Vec3(0, 1.5, 0), fov=55)
            else:
                cam = Camera(Vec3(8, 5, 8), Vec3(0, 1, 0), fov=55)

            img = renderer.render(scene, cam)
            frame_path = f"{basename}_{frame:04d}{ext}"
            renderer.save(img, frame_path)
        print(f"Animation complete: {args.animate} frames")
    else:
        # Single frame
        scene = build_scene(0.0)

        # Set up camera based on scene
        if args.scene == 'demo':
            cam = Camera(Vec3(7, 4, 7), Vec3(0, 1, 0), fov=55)
        elif args.scene == 'chess':
            cam = Camera(Vec3(5, 5, 8), Vec3(0, 0.5, 0), fov=50)
        elif args.scene == 'terrain':
            cam = Camera(Vec3(8, 5, 8), Vec3(0, 1, 0), fov=60)
        elif args.scene == 'abstract':
            cam = Camera(Vec3(6, 4, 6), Vec3(0, 1.5, 0), fov=55)
        else:
            cam = Camera(Vec3(7, 4, 7), Vec3(0, 1, 0), fov=55)

        print(f"Rendering scene '{args.scene}' ({args.width}x{args.height})")
        img = renderer.render(scene, cam)
        renderer.save(img, args.output)


if __name__ == '__main__':
    main()