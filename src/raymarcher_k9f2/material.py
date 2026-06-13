"""Material definitions for surface shading."""

from dataclasses import dataclass, field
from raymarcher_k9f2.vec3 import Vec3


@dataclass
class Material:
    """Surface material properties for ray-marched objects.

    Attributes:
        albedo: Base diffuse color.
        specular: Specular highlight strength (0–1).
        roughness: 0 = mirror, 1 = fully diffuse.
        reflectivity: 0 = no reflection, 1 = perfect mirror.
        emissive: Emissive light color (bypasses shading when non-zero).
        fresnel: Schlick Fresnel base reflectivity (typical: 0.04 for dielectrics).
        ior: Index of refraction for transmissive materials
            (1.0 = air, 1.33 = water, 1.5 = glass, 2.42 = diamond).
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

    def to_dict(self) -> dict:
        """Serialize material to a dictionary for config file support."""
        return {
            'albedo': [self.albedo.x, self.albedo.y, self.albedo.z],
            'specular': self.specular,
            'roughness': self.roughness,
            'reflectivity': self.reflectivity,
            'emissive': [self.emissive.x, self.emissive.y, self.emissive.z],
            'fresnel': self.fresnel,
            'ior': self.ior,
            'transparency': self.transparency,
            'subsurface': self.subsurface,
            'subsurface_color': [self.subsurface_color.x, self.subsurface_color.y, self.subsurface_color.z],
            'checker_scale': self.checker_scale,
            'checker_color': [self.checker_color.x, self.checker_color.y, self.checker_color.z],
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Material':
        """Deserialize material from a dictionary."""
        def _vec(val, default=None):
            if val is None:
                return Vec3(*default) if default else Vec3(0, 0, 0)
            return Vec3(*val)

        return cls(
            albedo=_vec(d.get('albedo'), [0.8, 0.8, 0.8]),
            specular=d.get('specular', 0.5),
            roughness=d.get('roughness', 0.3),
            reflectivity=d.get('reflectivity', 0.0),
            emissive=_vec(d.get('emissive'), [0, 0, 0]),
            fresnel=d.get('fresnel', 0.04),
            ior=d.get('ior', 1.5),
            transparency=d.get('transparency', 0.0),
            subsurface=d.get('subsurface', 0.0),
            subsurface_color=_vec(d.get('subsurface_color'), [0.5, 0.2, 0.1]),
            checker_scale=d.get('checker_scale', 0.0),
            checker_color=_vec(d.get('checker_color'), [0.2, 0.2, 0.2]),
        )


# ─── Preset Materials ──────────────────────────────────────────────────────

def red_plastic() -> Material:
    """Classic red plastic material."""
    return Material(albedo=Vec3(0.9, 0.2, 0.15), specular=0.5, roughness=0.3, reflectivity=0.1)

def blue_plastic() -> Material:
    """Blue plastic material."""
    return Material(albedo=Vec3(0.2, 0.5, 0.9), specular=0.5, roughness=0.2, reflectivity=0.1)

def mirror() -> Material:
    """Perfect mirror material."""
    return Material(albedo=Vec3(0.95, 0.95, 0.95), specular=1.0, roughness=0.0, reflectivity=0.95, fresnel=0.02)

def glass(ior: float = 1.5) -> Material:
    """Glass material with configurable index of refraction."""
    return Material(
        albedo=Vec3(0.95, 0.95, 0.98), specular=0.9, roughness=0.02,
        reflectivity=0.3, fresnel=0.04, transparency=0.9, ior=ior,
    )

def water() -> Material:
    """Water material (IOR=1.33)."""
    return Material(
        albedo=Vec3(0.7, 0.85, 0.95), specular=0.9, roughness=0.01,
        reflectivity=0.15, transparency=0.7, ior=1.33,
    )

def gold() -> Material:
    """Gold-like metallic material."""
    return Material(albedo=Vec3(1.0, 0.76, 0.33), specular=0.9, roughness=0.15, reflectivity=0.8, fresnel=0.5)

def checkerboard(scale: float = 1.0, light: Vec3 = None, dark: Vec3 = None) -> Material:
    """Checkerboard pattern material."""
    if light is None:
        light = Vec3(0.8, 0.8, 0.8)
    if dark is None:
        dark = Vec3(0.2, 0.2, 0.25)
    return Material(
        albedo=light, specular=0.2, roughness=0.7,
        reflectivity=0.15, checker_scale=scale, checker_color=dark,
    )