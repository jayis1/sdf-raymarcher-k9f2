"""Core ray marching engine with configurable quality settings."""

import math
import logging
from typing import Optional

from raymarcher_k9f2.vec3 import Vec3, BLACK
from raymarcher_k9f2.material import Material
from raymarcher_k9f2.patterns import checkerboard
from raymarcher_k9f2.scene import Scene, HitResult

logger = logging.getLogger(__name__)


class RayMarcher:
    """Core ray marching engine with configurable quality settings.

    Supports hard/soft shadows, ambient occlusion, reflections, refractions,
    subsurface scattering, and fog.

    Args:
        max_steps: Maximum ray march steps per ray.
        max_distance: Maximum ray travel distance before giving up.
        surface_epsilon: Distance threshold for a surface hit.
        normal_epsilon: Step size for central-difference normal estimation.
        max_bounces: Maximum number of reflection bounces.
        enable_shadows: Whether to compute shadows.
        enable_soft_shadows: Whether to use soft shadows (penumbra).
        shadow_softness: Soft shadow penumbra factor.
        enable_ao: Whether to compute ambient occlusion.
        ao_steps: Number of AO sampling steps.
        ao_step_size: Distance between AO samples.
        ao_strength: AO darkening strength.
        max_refraction_bounces: Maximum refraction bounces.
    """

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

            t += max(d, self.surface_epsilon * 0.5)
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