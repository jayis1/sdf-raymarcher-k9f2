#!/usr/bin/env python3
"""
SDF Ray Marcher k9f2 — A pure-Python signed distance field ray marcher.

Renders 3D scenes using ray marching against analytic signed distance fields.
Supports multiple primitives, materials, soft shadows, ambient occlusion,
reflections, refractions (glass), subsurface scattering, depth-of-field,
and animation.

Usage:
    source venv/bin/activate
    python3 raymarcher.py [options]

Options:
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
"""

import math
import argparse
import os
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
        """Component-wise multiplication if other is Vec3, scalar otherwise."""
        if isinstance(other, Vec3):
            return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        return Vec3(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, scalar: float) -> 'Vec3':
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

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

    def length_sq(self) -> float:
        """Squared length — avoids sqrt when comparing distances."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalized(self) -> 'Vec3':
        l = self.length()
        if l < 1e-12:
            return Vec3(0, 0, 0)
        inv = 1.0 / l
        return Vec3(self.x * inv, self.y * inv, self.z * inv)

    def reflect(self, normal: 'Vec3') -> 'Vec3':
        """Reflect this vector off a surface with given normal."""
        return self - normal * (2.0 * self.dot(normal))

    def refract(self, normal: 'Vec3', eta: float) -> Optional['Vec3']:
        """Refract this vector through a surface. Returns None for total internal reflection."""
        cos_i = -(self.dot(normal))
        if cos_i < 0:
            # We're inside the object; flip normal and eta
            cos_i = -cos_i
            normal = -normal
            eta = 1.0 / eta
        sin_t2 = eta * eta * (1.0 - cos_i * cos_i)
        if sin_t2 > 1.0:
            return None  # Total internal reflection
        cos_t = math.sqrt(1.0 - sin_t2)
        return self * eta + normal * (eta * cos_i - cos_t)

    def lerp(self, other: 'Vec3', t: float) -> 'Vec3':
        return self * (1.0 - t) + other * t

    def clamp(self, lo: float = 0.0, hi: float = 1.0) -> 'Vec3':
        return Vec3(
            max(lo, min(hi, self.x)),
            max(lo, min(hi, self.y)),
            max(lo, min(hi, self.z)),
        )

    def pow(self, exponent: float) -> 'Vec3':
        """Component-wise power."""
        return Vec3(
            pow(max(self.x, 0.0), exponent),
            pow(max(self.y, 0.0), exponent),
            pow(max(self.z, 0.0), exponent),
        )


# ─── Color Constants ────────────────────────────────────────────────────────

BLACK = Vec3(0, 0, 0)
WHITE = Vec3(1, 1, 1)
SKY_BLUE = Vec3(0.5, 0.7, 1.0)


# ─── Material ────────────────────────────────────────────────────────────────

@dataclass
class Material:
    """Surface material properties.

    Attributes:
        albedo: Base diffuse color.
        specular: Specular highlight strength (0–1).
        roughness: 0 = mirror, 1 = fully diffuse.
        reflectivity: 0 = no reflection, 1 = perfect mirror.
        emissive: Emissive light color (bypasses shading when non-zero).
        fresnel: Schlick Fresnel base reflectivity (typical: 0.04 for dielectrics).
        ior: Index of refraction for transmissive materials (1.0 = air, 1.5 = glass, 2.42 = diamond).
        transparency: 0 = opaque, 1 = fully transparent. Enables refraction when > 0.
        subsurface: Subsurface scattering approximation strength (0 = none).
        subsurface_color: Color tint for subsurface scattering.
        checker_scale: If > 0, applies a procedural checkerboard pattern at this scale.
        checker_color: Secondary checkerboard color.
    """
    albedo: Vec3 = field(default_factory=lambda: Vec3(0.8, 0.8, 0.8))
    specular: float = 0.5
    roughness: float = 0.3
    reflectivity: float = 0.0
    emissive: Vec3 = field(default_factory=lambda: Vec3(0, 0, 0))
    fresnel: float = 0.04
    ior: float = 1.5
    transparency: float = 0.0
    subsurface: float = 0.0
    subsurface_color: Vec3 = field(default_factory=lambda: Vec3(0.5, 0.2, 0.1))
    checker_scale: float = 0.0
    checker_color: Vec3 = field(default_factory=lambda: Vec3(0.2, 0.2, 0.2))


# ─── SDF Primitives ─────────────────────────────────────────────────────────

def sdf_sphere(p: Vec3, center: Vec3, radius: float) -> float:
    """Signed distance to a sphere."""
    return (p - center).length() - radius


def sdf_box(p: Vec3, center: Vec3, half_extents: Vec3) -> float:
    """Signed distance to an axis-aligned box.

    Uses the standard SDF formulation:
    - Outside: Euclidean distance to nearest face/edge/corner
    - Inside: negative of the minimum distance to a face (closest face depth)
    """
    q = p - center
    qx, qy, qz = abs(q.x), abs(q.y), abs(q.z)
    # Outside: distance to nearest point on box surface
    cx = max(qx - half_extents.x, 0.0)
    cy = max(qy - half_extents.y, 0.0)
    cz = max(qz - half_extents.z, 0.0)
    outside_dist = math.sqrt(cx * cx + cy * cy + cz * cz)
    # Inside: negative of how deep we are (minimum face distance)
    inside_dist = min(half_extents.x - qx, half_extents.y - qy, half_extents.z - qz)
    return outside_dist if inside_dist < 0 else -inside_dist


def sdf_torus(p: Vec3, center: Vec3, major_r: float, minor_r: float) -> float:
    """Signed distance to a torus in the XZ plane."""
    q = p - center
    xz_dist = math.sqrt(q.x * q.x + q.z * q.z) - major_r
    return math.sqrt(xz_dist * xz_dist + q.y * q.y) - minor_r


def sdf_cylinder(p: Vec3, center: Vec3, radius: float, half_height: float) -> float:
    """Signed distance to a vertical cylinder."""
    q = p - center
    dist_xz = math.sqrt(q.x * q.x + q.z * q.z) - radius
    dist_y = abs(q.y) - half_height
    outside = math.sqrt(max(dist_xz, 0.0) ** 2 + max(dist_y, 0.0) ** 2)
    inside = min(max(dist_xz, dist_y), 0.0)
    return outside + inside


def sdf_capsule(p: Vec3, a: Vec3, b: Vec3, radius: float) -> float:
    """Signed distance to a capsule (line-segment swept sphere)."""
    ab = b - a
    ap = p - a
    t = ap.dot(ab) / max(ab.dot(ab), 1e-12)  # Guard against zero-length segment
    t = max(0.0, min(1.0, t))
    closest = a + ab * t
    return (p - closest).length() - radius


def sdf_plane(p: Vec3, height: float = 0.0) -> float:
    """Signed distance to a horizontal plane at given height."""
    return p.y - height


def sdf_cone(p: Vec3, center: Vec3, half_angle: float, height: float) -> float:
    """Signed distance to a cone pointing up from center.y with base at center.y.

    The cone has its tip at (center.x, center.y + height, center.z) and
    base radius = height * tan(half_angle) at center.y.

    Args:
        p: Point to evaluate.
        center: Center of the cone base.
        half_angle: Half-angle of the cone in radians.
        height: Height of the cone.
    """
    q = p - center
    # Cone without cap: distance to infinite double cone
    sin_a = math.sin(half_angle)
    cos_a = math.cos(half_angle)
    # Distance from point to cone surface
    # Project onto 2D (distance from axis, height)
    d_xz = math.sqrt(q.x * q.x + q.z * q.z)
    # Cone surface equation: d_xz = (height - q.y) * tan(half_angle)
    # SDF: cos_a * d_xz - sin_a * (height - q.y) for the cone surface
    cone_dist = cos_a * d_xz + sin_a * (q.y - height)

    # Cap at base (y = 0 relative to center)
    base_dist = -q.y

    # Cap at the bottom
    if q.y < 0:
        # Below base: distance is base_dist or cone_dist, whichever is closer
        return math.sqrt(d_xz * d_xz + q.y * q.y)

    # Intersect cone with base plane
    return max(cone_dist, base_dist)


def sdf_hex_prism(p: Vec3, center: Vec3, radius: float, half_height: float) -> float:
    """Signed distance to a hexagonal prism (vertical, in XZ plane).

    Uses the intersection of six half-planes for the hexagonal cross-section,
    extruded along Y.
    """
    q = p - center
    k = math.sqrt(3.0) * 0.5
    qx_abs = abs(q.x)
    qz_abs = abs(q.z)
    # The three pairs of half-planes that define a regular hexagon
    d2 = qx_abs - radius
    d3 = qz_abs - radius * k
    d4 = qx_abs * 0.5 + qz_abs * k - radius
    d_hex = max(d2, max(d3, d4))
    # Extrude in Y
    d_y = abs(q.y) - half_height
    outside = math.sqrt(max(d_hex, 0.0) ** 2 + max(d_y, 0.0) ** 2)
    inside = min(max(d_hex, d_y), 0.0)
    return outside + inside


def sdf_mandelbulb(p: Vec3, center: Vec3, scale: float = 1.0, power: float = 8.0,
                   iterations: int = 8) -> float:
    """Approximate distance estimator for the Mandelbulb fractal.

    This is a ray-marchable distance estimator, not an exact SDF —
    we use a normalized orbit trap method.
    """
    z = p - center
    z = z * scale  # Scale to reasonable size
    dr = 1.0
    r = 0.0

    for _ in range(iterations):
        r = z.length()
        if r > 2.0:
            break
        # Convert to spherical coordinates
        theta = math.acos(max(-1.0, min(1.0, z.z / max(r, 1e-12))))
        phi = math.atan2(z.y, z.x)
        dr = pow(r, power - 1.0) * power * dr + 1.0

        # New z = r^power * (sin(theta*power), sin(phi*power), cos(theta*power))
        zr = pow(r, power)
        theta *= power
        phi *= power
        z = Vec3(
            zr * math.sin(theta) * math.cos(phi),
            zr * math.sin(theta) * math.sin(phi),
            zr * math.cos(theta),
        ) + (p - center) * scale

    return 0.5 * math.log(max(r, 1e-12)) * r / max(dr, 1.0) / scale


# ─── CSG Operations ──────────────────────────────────────────────────────────

def sdf_union(d1: float, d2: float) -> float:
    """Boolean union (min) of two SDF values."""
    return min(d1, d2)


def sdf_smooth_union(d1: float, d2: float, k: float = 0.5) -> float:
    """Smooth boolean union with blending factor k."""
    h = max(k - abs(d1 - d2), 0.0) / k
    return min(d1, d2) - h * h * k * 0.25


def sdf_smooth_subtraction(d1: float, d2: float, k: float = 0.5) -> float:
    """Smooth subtraction: d1 minus d2, with smooth blending factor k."""
    h = max(k - abs(-d2 - d1), 0.0) / k
    return max(-d2, d1) + h * h * k * 0.25


def sdf_subtraction(d1: float, d2: float) -> float:
    """Subtract d2 from d1 (d1 - d2)."""
    return max(-d2, d1)


def sdf_intersection(d1: float, d2: float) -> float:
    """Boolean intersection of two SDF values."""
    return max(d1, d2)


# ─── Procedural Patterns ─────────────────────────────────────────────────────

def checkerboard(p: Vec3, scale: float, color1: Vec3, color2: Vec3) -> Vec3:
    """3D checkerboard pattern at given scale."""
    ix = math.floor(p.x * scale)
    iy = math.floor(p.y * scale)
    iz = math.floor(p.z * scale)
    if (ix + iy + iz) % 2 == 0:
        return color1
    return color2


# ─── Scene Objects ───────────────────────────────────────────────────────────

@dataclass
class SceneObject:
    """An object in the scene: SDF function + material."""
    sdf_func: Callable[[Vec3], float]
    material: Material
    interior_material: Optional[Material] = None  # For refractive objects (inside vs outside)


# ─── Hit Info ────────────────────────────────────────────────────────────────

@dataclass
class HitResult:
    distance: float = float('inf')
    material: Optional[Material] = None
    object_index: int = -1
    entering: bool = True  # True if entering, False if exiting (for refraction)


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
        self.environment_color: Vec3 = Vec3(0.4, 0.5, 0.6)  # Fallback for refractive rays

    def add_object(self, sdf_func: Callable[[Vec3], float], material: Material,
                   interior_material: Material = None) -> int:
        idx = len(self.objects)
        self.objects.append(SceneObject(sdf_func, material, interior_material))
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
                 ao_strength: float = 1.0, max_refraction_bounces: int = 5):
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
        self.max_refraction_bounces = max_refraction_bounces

    def march(self, origin: Vec3, direction: Vec3, scene: Scene) -> HitResult:
        """March a ray through the scene, returning the closest hit.

        For refractive objects, we detect whether we are entering or exiting
        by checking the SDF at the ray origin — if it's negative, we're
        already inside an object (exiting).
        """
        t = 0.0
        hit = HitResult()

        # Check if we're starting inside an object (for refraction)
        initial_result = scene.map(origin)
        hit.entering = initial_result.distance >= 0  # Outside = entering

        for step in range(self.max_steps):
            p = origin + direction * t
            result = scene.map(p)
            d = result.distance

            if d < self.surface_epsilon:
                hit.distance = t
                hit.material = result.material
                hit.object_index = result.object_index
                return hit

            t += d

            if t > self.max_distance:
                break

        hit.distance = t
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
                penumbra = max(d / max(t, 1e-6), 0.0)
                shadow = min(shadow, self.shadow_softness * penumbra)
            else:
                if d < self.surface_epsilon:
                    return 0.0  # Hard shadow: fully occluded

            t += max(d, self.surface_epsilon * 0.5)  # Prevent getting stuck
            if t > self.max_distance:
                break

        return max(0.0, min(1.0, shadow))

    def _get_albedo(self, material: Material, p: Vec3) -> Vec3:
        """Get the effective albedo, considering checkerboard patterns."""
        if material.checker_scale > 0:
            return checkerboard(p, material.checker_scale, material.albedo, material.checker_color)
        return material.albedo

    def shade(self, p: Vec3, normal: Vec3, view_dir: Vec3,
              material: Material, scene: Scene) -> Vec3:
        """Shade a surface point with all lighting contributions."""
        # Emissive materials bypass shading
        if material.emissive.length() > 0:
            return material.emissive

        color = self._get_albedo(material, p)
        result = BLACK

        # Ambient
        ambient_strength = 0.15
        if self.enable_ao:
            ambient_strength *= self.compute_ao(p, normal, scene)
        result = result + color * ambient_strength

        # Subsurface scattering approximation
        if material.subsurface > 0:
            sss = self._compute_subsurface(p, normal, material, scene)
            result = result + sss

        # Directional lights
        for light in scene.directional_lights:
            light_dir = light.direction
            n_dot_l = max(normal.dot(light_dir), 0.0)

            shadow = self.compute_shadow(p, light_dir, scene)

            # Diffuse
            diffuse = color * (n_dot_l * light.color * light.intensity * shadow)

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
            if dist < 1e-6:
                continue
            light_dir = to_light * (1.0 / dist)
            attenuation = 1.0 / (1.0 + 0.09 * dist + 0.032 * dist * dist)
            n_dot_l = max(normal.dot(light_dir), 0.0)

            shadow = self.compute_shadow(p, light_dir, scene)

            diffuse = color * (n_dot_l * light.color * light.intensity * attenuation * shadow)

            half_vec = (light_dir + view_dir).normalized()
            n_dot_h = max(normal.dot(half_vec), 0.0)
            spec_pow = max(1.0, (1.0 - material.roughness) * 256.0)
            specular = Vec3(1, 1, 1) * (n_dot_h ** spec_pow) * material.specular * shadow

            result = result + diffuse + specular

        return result.clamp(0.0, 10.0)

    def _compute_subsurface(self, p: Vec3, normal: Vec3, material: Material,
                            scene: Scene) -> Vec3:
        """Approximate subsurface scattering by sampling light through the object."""
        sss = BLACK
        strength = material.subsurface

        for light in scene.directional_lights:
            # Estimate how much light comes through from the back side
            back_n_dot_l = max((-normal).dot(light.direction), 0.0)
            sss = sss + material.subsurface_color * back_n_dot_l * light.color * light.intensity * strength

        for light in scene.point_lights:
            to_light = light.position - p
            dist = to_light.length()
            if dist < 1e-6:
                continue
            light_dir = to_light * (1.0 / dist)
            attenuation = 1.0 / (1.0 + 0.09 * dist + 0.032 * dist * dist)
            back_n_dot_l = max((-normal).dot(light_dir), 0.0)
            sss = sss + material.subsurface_color * back_n_dot_l * light.color * light.intensity * attenuation * strength

        return sss

    def trace(self, origin: Vec3, direction: Vec3, scene: Scene,
              bounce: int = 0, refraction_depth: int = 0) -> Vec3:
        """Trace a single ray, handling reflections and refractions."""
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
            reflect_color = self.trace(reflect_origin, reflect_dir, scene, bounce + 1, refraction_depth)
            cos_theta = max(view_dir.dot(normal), 0.0)
            fresnel = hit.material.fresnel + (1.0 - hit.material.fresnel) * ((1.0 - cos_theta) ** 5)
            reflect_amount = max(hit.material.reflectivity, fresnel) * (1.0 - hit.material.roughness)
            color = color * (1.0 - reflect_amount) + reflect_color * reflect_amount

        # Refractions (glass/water)
        if hit.material.transparency > 0 and refraction_depth < self.max_refraction_bounces:
            cos_i = max(-direction.dot(normal), 0.0)
            # Determine if entering or exiting
            ior = hit.material.ior
            if not hit.entering:
                # Exiting glass: going from glass (IOR medium) to air
                # Snell's law: n1*sin(θ1) = n2*sin(θ2), eta = n1/n2 = ior/1.0
                eta = ior
            else:
                eta = 1.0 / ior  # air to glass

            refract_dir = direction.refract(normal, eta)
            if refract_dir is not None:
                refract_origin = p - normal * self.surface_epsilon * 2.0
                refract_color = self.trace(refract_origin, refract_dir, scene, bounce, refraction_depth + 1)

                # Fresnel for glass (Schlick)
                cos_theta = cos_i
                r0 = ((1.0 - ior) / (1.0 + ior)) ** 2
                fresnel = r0 + (1.0 - r0) * ((1.0 - cos_theta) ** 5)
                transmittance = hit.material.transparency * (1.0 - fresnel)

                # Beer's law: tint the refracted light based on distance
                absorption = hit.material.albedo * hit.material.transparency
                refract_color = refract_color * absorption.clamp(0.0, 1.0)

                color = color * (1.0 - hit.material.transparency) + refract_color * transmittance
            else:
                # Total internal reflection
                reflect_dir = direction.reflect(normal)
                reflect_origin = p + normal * self.surface_epsilon * 2.0
                if bounce < self.max_bounces:
                    tir_color = self.trace(reflect_origin, reflect_dir, scene, bounce + 1, refraction_depth)
                    color = color * (1.0 - hit.material.transparency) + tir_color * hit.material.transparency

        # Fog
        if scene.fog_density > 0:
            fog_factor = 1.0 - math.exp(-scene.fog_density * hit.distance)
            color = color.lerp(scene.fog_color, fog_factor)

        return color.clamp(0.0, 1.0)


# ─── Camera ──────────────────────────────────────────────────────────────────

class Camera:
    """Perspective camera with look-at targeting and optional depth-of-field."""

    def __init__(self, position: Vec3, target: Vec3, fov: float = 60.0,
                 up: Vec3 = None, focal_distance: float = 0.0, aperture: float = 0.1):
        if up is None:
            up = Vec3(0, 1, 0)
        self.position = position
        self.target = target
        self.fov = fov
        self.up = up
        self.focal_distance = focal_distance  # 0 = no DOF
        self.aperture = aperture

    def get_ray(self, u: float, v: float, aspect: float = 1.0,
                sample_index: int = 0) -> Tuple[Vec3, Vec3]:
        """Get ray for normalized screen coordinates u,v in [-0.5, 0.5].

        Args:
            u: Horizontal screen coordinate, -0.5 (left) to 0.5 (right).
            v: Vertical screen coordinate, -0.5 (bottom) to 0.5 (top).
            aspect: Width/height aspect ratio.
            sample_index: Sample index for DOF disk sampling.

        When focal_distance > 0, applies thin-lens DOF with stratified
        disk sampling based on sample_index.
        """
        forward = (self.target - self.position).normalized()
        right = forward.cross(self.up).normalized()
        true_up = right.cross(forward).normalized()

        fov_rad = math.radians(self.fov)
        half_h = math.tan(fov_rad / 2.0)
        half_w = half_h * aspect

        # Direction toward focal point on the image plane
        direction = (forward + right * (u * 2.0 * half_w) + true_up * (v * 2.0 * half_h)).normalized()

        origin = self.position

        if self.focal_distance > 0 and self.aperture > 0:
            # Thin lens model: jitter the origin on a disk, then
            # redirect direction toward the focal plane point
            focal_point = origin + direction * self.focal_distance

            # Stratified disk sampling using golden ratio
            angle = sample_index * 2.399963  # golden angle in radians
            r = math.sqrt((sample_index % 16 + 0.5) / 16.0) * self.aperture
            jitter_x = math.cos(angle) * r
            jitter_y = math.sin(angle) * r

            origin = self.position + right * jitter_x + true_up * jitter_y
            direction = (focal_point - origin).normalized()

        return origin, direction


# ─── Renderer ────────────────────────────────────────────────────────────────

class Renderer:
    """Turns scenes into images using the ray marcher."""

    def __init__(self, width: int = 800, height: int = 600, samples: int = 1,
                 marcher: RayMarcher = None, quiet: bool = False):
        self.width = width
        self.height = height
        self.samples = samples
        self.marcher = marcher or RayMarcher()
        self.quiet = quiet

    def render(self, scene: Scene, camera: Camera) -> np.ndarray:
        """Render the scene, returning an (H, W, 3) float array."""
        img = np.zeros((self.height, self.width, 3), dtype=np.float64)
        aspect = self.width / self.height

        total_pixels = self.width * self.height
        last_pct = -1
        t_start = time.time()

        for y in range(self.height):
            pct = int((y * self.width) / total_pixels * 100)
            if pct != last_pct and pct % 10 == 0 and not self.quiet:
                elapsed = time.time() - t_start
                remaining = (elapsed / max(pct, 1)) * (100 - pct) if pct > 0 else 0
                last_pct = pct
                print(f"  Rendering: {pct}% ({elapsed:.1f}s, ~{remaining:.0f}s remaining)")

            for x in range(self.width):
                color = BLACK
                for s in range(self.samples):
                    # Jittered sampling with golden ratio for better distribution
                    if self.samples > 1:
                        jx = (x + (s * 0.618033988749895 % 1.0)) / self.width
                        jy = (y + (s * 0.414213562373095 % 1.0)) / self.height
                    else:
                        jx = (x + 0.5) / self.width
                        jy = (y + 0.5) / self.height

                    # Normalized coordinates in [-0.5, 0.5]
                    u = jx - 0.5
                    v = 0.5 - jy  # Flip Y so up is positive

                    origin, direction = camera.get_ray(u, v, aspect, s)
                    sample_color = self.marcher.trace(origin, direction, scene)
                    color = color + sample_color

                color = color * (1.0 / self.samples)
                # Gamma correction
                color = color.pow(1.0 / 2.2)
                img[y, x, 0] = color.x
                img[y, x, 1] = color.y
                img[y, x, 2] = color.z

        elapsed = time.time() - t_start
        if not self.quiet:
            print(f"  Rendering: 100% ({elapsed:.1f}s)")
        return img

    def save(self, img: np.ndarray, path: str):
        """Save float image array as 8-bit PNG."""
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, 'RGB')
        pil_img.save(path)
        if not self.quiet:
            print(f"  Saved: {path} ({img.shape[1]}x{img.shape[0]})")


# ─── Scene Builders ──────────────────────────────────────────────────────────

def build_demo_scene(time_val: float = 0.0) -> Scene:
    """Default demo scene with various primitives."""
    scene = Scene()
    scene.time = time_val

    # Ground plane - checkerboard
    ground_mat = Material(
        albedo=Vec3(0.7, 0.7, 0.7),
        specular=0.2,
        roughness=0.7,
        reflectivity=0.15,
        checker_scale=1.0,
        checker_color=Vec3(0.3, 0.3, 0.35),
    )

    def ground_sdf(p: Vec3) -> float:
        return sdf_plane(p, height=0.0)

    scene.add_object(ground_sdf, ground_mat)

    # Animated sphere (red, reflective)
    def sphere_sdf(p: Vec3) -> float:
        center = Vec3(0, 1.0 + 0.3 * math.sin(time_val * 1.5), 0)
        return sdf_sphere(p, center, 1.0)

    sphere_mat = Material(
        albedo=Vec3(0.9, 0.2, 0.15),
        specular=0.9,
        roughness=0.05,
        reflectivity=0.4,
        fresnel=0.05,
    )
    scene.add_object(sphere_sdf, sphere_mat)

    # Box (blue)
    def box_sdf(p: Vec3) -> float:
        return sdf_box(p, Vec3(3.0, 0.75, 1.0), Vec3(0.75, 0.75, 0.75))

    box_mat = Material(
        albedo=Vec3(0.2, 0.5, 0.9),
        specular=0.6,
        roughness=0.2,
        reflectivity=0.1,
    )
    scene.add_object(box_sdf, box_mat)

    # Torus (yellow)
    def torus_sdf(p: Vec3) -> float:
        return sdf_torus(p, Vec3(-2.5, 0.8, 2.0), 0.8, 0.25)

    torus_mat = Material(
        albedo=Vec3(0.9, 0.85, 0.1),
        specular=0.7,
        roughness=0.15,
    )
    scene.add_object(torus_sdf, torus_mat)

    # Capsule (green)
    def capsule_sdf(p: Vec3) -> float:
        return sdf_capsule(p, Vec3(-3.0, 0.5, -1.5), Vec3(-3.0, 2.5, -1.5), 0.35)

    capsule_mat = Material(
        albedo=Vec3(0.1, 0.8, 0.4),
        specular=0.5,
        roughness=0.3,
    )
    scene.add_object(capsule_sdf, capsule_mat)

    # Smooth union of two spheres (orange)
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

    # Cylinder (purple)
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


def build_chess_scene(time_val: float = 0.0) -> Scene:
    """Chess-like scene with a reflective board and pieces."""
    scene = Scene()
    scene.time = time_val
    scene.fog_density = 0.015
    scene.fog_color = Vec3(0.35, 0.3, 0.25)

    # Board
    board_mat = Material(albedo=Vec3(0.15, 0.12, 0.1), specular=0.3, roughness=0.6, reflectivity=0.3)

    def board_sdf(p: Vec3) -> float:
        return sdf_box(p, Vec3(0, -0.1, 0), Vec3(5, 0.1, 5))

    scene.add_object(board_sdf, board_mat)

    # Checker pattern
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

    # King
    king_mat = Material(albedo=Vec3(0.95, 0.92, 0.85), specular=0.8, roughness=0.1, reflectivity=0.2)

    def king_sdf(p: Vec3) -> float:
        body = sdf_cylinder(p, Vec3(0, 0.7, 0), 0.25, 0.7)
        head = sdf_sphere(p, Vec3(0, 1.6, 0), 0.22)
        base = sdf_cylinder(p, Vec3(0, 0.15, 0), 0.35, 0.15)
        return sdf_smooth_union(sdf_smooth_union(body, head, k=0.15), base, k=0.1)

    scene.add_object(king_sdf, king_mat)

    # Pawn
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


def build_terrain_scene(time_val: float = 0.0) -> Scene:
    """Procedural terrain scene with water."""
    scene = Scene()
    scene.time = time_val
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
        # Gentle wave animation
        wave = math.sin(p.x * 2.0 + time_val) * 0.02 + math.cos(p.z * 3.0 + time_val * 0.7) * 0.015
        return sdf_plane(p, height=0.3 + wave)

    water_mat = Material(
        albedo=Vec3(0.15, 0.3, 0.6),
        specular=0.9,
        roughness=0.05,
        reflectivity=0.6,
        transparency=0.3,
        ior=1.33,  # Water
    )
    scene.add_object(water_sdf, water_mat)

    scene.add_directional_light(Vec3(0.3, 0.8, 0.5), Vec3(1.0, 0.9, 0.7), 1.0)
    scene.add_directional_light(Vec3(-0.5, 0.3, -0.3), Vec3(0.4, 0.5, 0.7), 0.3)

    return scene


def build_abstract_scene(time_val: float = 0.0) -> Scene:
    """Abstract scene with morphing smooth unions."""
    scene = Scene()
    scene.time = time_val

    def abstract_sdf(p: Vec3) -> float:
        d = float('inf')
        for i in range(5):
            angle = time_val * 0.5 + i * math.pi * 2.0 / 5.0
            cx = 2.0 * math.cos(angle)
            cy = 1.5 + 0.5 * math.sin(time_val + i)
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

    ground_mat = Material(albedo=Vec3(0.2, 0.2, 0.25), specular=0.2, roughness=0.8, reflectivity=0.3)
    scene.add_object(lambda p: sdf_plane(p, height=0.0), ground_mat)

    scene.add_directional_light(Vec3(0.5, 0.8, 0.3), Vec3(1.0, 0.95, 0.85), 1.0)
    scene.add_directional_light(Vec3(-0.3, 0.5, -0.7), Vec3(0.5, 0.3, 0.7), 0.4)

    return scene


def build_glass_scene(time_val: float = 0.0) -> Scene:
    """Scene showcasing glass/refraction and subsurface scattering."""
    scene = Scene()
    scene.time = time_val

    # Ground plane - checkerboard
    ground_mat = Material(
        albedo=Vec3(0.8, 0.8, 0.8),
        specular=0.2,
        roughness=0.7,
        reflectivity=0.2,
        checker_scale=2.0,
        checker_color=Vec3(0.3, 0.3, 0.3),
    )
    scene.add_object(lambda p: sdf_plane(p, height=0.0), ground_mat)

    # Glass sphere (center)
    glass_mat = Material(
        albedo=Vec3(0.95, 0.95, 0.98),
        specular=0.9,
        roughness=0.02,
        reflectivity=0.3,
        fresnel=0.04,
        transparency=0.9,
        ior=1.5,  # Glass
    )
    scene.add_object(lambda p: sdf_sphere(p, Vec3(0, 1.2, 0), 1.2), glass_mat)

    # Subsurface scattering sphere (wax/candle-like)
    sss_mat = Material(
        albedo=Vec3(0.9, 0.7, 0.5),
        specular=0.3,
        roughness=0.6,
        subsurface=0.8,
        subsurface_color=Vec3(0.9, 0.4, 0.1),
    )
    scene.add_object(lambda p: sdf_sphere(p, Vec3(-3, 0.8, 1), 0.8), sss_mat)

    # Glass cube
    glass_cube_mat = Material(
        albedo=Vec3(0.95, 0.95, 0.97),
        specular=0.8,
        roughness=0.02,
        reflectivity=0.2,
        fresnel=0.04,
        transparency=0.85,
        ior=1.45,
    )
    scene.add_object(lambda p: sdf_box(p, Vec3(3, 0.75, -0.5), Vec3(0.75, 0.75, 0.75)), glass_cube_mat)

    # Opaque red sphere for contrast
    red_mat = Material(
        albedo=Vec3(0.9, 0.1, 0.1),
        specular=0.7,
        roughness=0.15,
        reflectivity=0.3,
    )
    scene.add_object(lambda p: sdf_sphere(p, Vec3(1.5, 0.6, 2.0), 0.6), red_mat)

    # Water IOR sphere
    water_mat = Material(
        albedo=Vec3(0.7, 0.85, 0.95),
        specular=0.9,
        roughness=0.01,
        reflectivity=0.15,
        transparency=0.7,
        ior=1.33,
    )
    scene.add_object(lambda p: sdf_sphere(p, Vec3(-1.5, 0.7, -2.5), 0.7), water_mat)

    scene.add_directional_light(Vec3(0.5, 0.8, 0.3), Vec3(1.0, 0.95, 0.85), 1.2)
    scene.add_directional_light(Vec3(-0.3, 0.4, -0.5), Vec3(0.3, 0.4, 0.6), 0.3)
    scene.add_point_light(Vec3(-2, 3, 2), Vec3(1.0, 0.7, 0.3), 4.0)

    return scene


def build_fractal_scene(time_val: float = 0.0) -> Scene:
    """Mandelbulb fractal scene."""
    scene = Scene()
    scene.time = time_val
    scene.fog_density = 0.01
    scene.fog_color = Vec3(0.1, 0.1, 0.15)

    # Mandelbulb
    power = 8.0 + 2.0 * math.sin(time_val * 0.3)

    def fractal_sdf(p: Vec3) -> float:
        return sdf_mandelbulb(p, Vec3(0, 1.5, 0), scale=1.2, power=power, iterations=8)

    fractal_mat = Material(
        albedo=Vec3(0.8, 0.6, 0.3),
        specular=0.5,
        roughness=0.4,
        reflectivity=0.1,
    )
    scene.add_object(fractal_sdf, fractal_mat)

    # Ground
    ground_mat = Material(
        albedo=Vec3(0.15, 0.15, 0.2),
        specular=0.3,
        roughness=0.8,
        reflectivity=0.3,
    )
    scene.add_object(lambda p: sdf_plane(p, height=0.0), ground_mat)

    scene.add_directional_light(Vec3(0.5, 0.8, 0.3), Vec3(1.0, 0.95, 0.85), 1.0)
    scene.add_directional_light(Vec3(-0.5, 0.3, -0.5), Vec3(0.4, 0.3, 0.6), 0.4)
    scene.add_point_light(Vec3(3, 4, -2), Vec3(0.8, 0.6, 1.0), 5.0)

    return scene


SCENES = {
    'demo': build_demo_scene,
    'chess': build_chess_scene,
    'terrain': build_terrain_scene,
    'abstract': build_abstract_scene,
    'glass': build_glass_scene,
    'fractal': build_fractal_scene,
}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='SDF Ray Marcher k9f2')
    parser.add_argument('--width', type=int, default=800, help='Image width')
    parser.add_argument('--height', type=int, default=600, help='Image height')
    parser.add_argument('--samples', type=int, default=1, help='Samples per pixel (AA)')
    parser.add_argument('--output', type=str, default='output.png', help='Output file path')
    parser.add_argument('--scene', type=str, default='demo', choices=SCENES.keys(), help='Scene to render')
    parser.add_argument('--animate', type=int, default=1, help='Number of animation frames')
    parser.add_argument('--max-bounces', type=int, default=3, help='Max reflection/refraction bounces')
    parser.add_argument('--ao', action='store_true', help='Enable ambient occlusion')
    parser.add_argument('--soft-shadow', action='store_true', help='Enable soft shadows')
    parser.add_argument('--no-shadow', action='store_true', help='Disable shadows')
    parser.add_argument('--dof', type=float, default=0, help='Depth of field focal distance (0=off)')
    parser.add_argument('--aperture', type=float, default=0.1, help='DOF aperture size')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')
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
        quiet=args.quiet,
    )

    build_scene = SCENES[args.scene]

    # Camera presets per scene
    def make_camera(scene_name: str, anim_t: float = 0.0) -> Camera:
        if scene_name == 'demo':
            if anim_t != 0:
                pos = Vec3(7 * math.cos(anim_t * 0.3), 4, 7 * math.sin(anim_t * 0.3))
            else:
                pos = Vec3(7, 4, 7)
            return Camera(pos, Vec3(0, 1, 0), fov=55, focal_distance=args.dof, aperture=args.aperture)
        elif scene_name == 'chess':
            return Camera(Vec3(5, 5, 8), Vec3(0, 0.5, 0), fov=50, focal_distance=args.dof or 0, aperture=args.aperture)
        elif scene_name == 'terrain':
            return Camera(Vec3(8, 5, 8), Vec3(0, 1, 0), fov=60, focal_distance=args.dof or 0, aperture=args.aperture)
        elif scene_name == 'abstract':
            if anim_t != 0:
                pos = Vec3(6 * math.cos(anim_t * 0.2), 4, 6 * math.sin(anim_t * 0.2))
            else:
                pos = Vec3(6, 4, 6)
            return Camera(pos, Vec3(0, 1.5, 0), fov=55, focal_distance=args.dof or 0, aperture=args.aperture)
        elif scene_name == 'glass':
            return Camera(Vec3(5, 3.5, 8), Vec3(0, 1, 0), fov=50, focal_distance=args.dof or 8, aperture=args.aperture)
        elif scene_name == 'fractal':
            if anim_t != 0:
                pos = Vec3(4 * math.cos(anim_t * 0.2), 3, 4 * math.sin(anim_t * 0.2))
            else:
                pos = Vec3(4, 3, 4)
            return Camera(pos, Vec3(0, 1.5, 0), fov=50, focal_distance=args.dof or 0, aperture=args.aperture)
        else:
            return Camera(Vec3(7, 4, 7), Vec3(0, 1, 0), fov=55, focal_distance=args.dof, aperture=args.aperture)

    if args.animate > 1:
        basename = os.path.splitext(args.output)[0]
        ext = os.path.splitext(args.output)[1] or '.png'
        for frame in range(args.animate):
            t = frame / args.animate * math.pi * 2.0
            if not args.quiet:
                print(f"Frame {frame + 1}/{args.animate}")
            scene = build_scene(t)
            cam = make_camera(args.scene, t)
            img = renderer.render(scene, cam)
            frame_path = f"{basename}_{frame:04d}{ext}"
            renderer.save(img, frame_path)
        if not args.quiet:
            print(f"Animation complete: {args.animate} frames")
    else:
        scene = build_scene(0.0)
        cam = make_camera(args.scene)

        if not args.quiet:
            print(f"Rendering scene '{args.scene}' ({args.width}x{args.height})")
        img = renderer.render(scene, cam)
        renderer.save(img, args.output)


if __name__ == '__main__':
    main()