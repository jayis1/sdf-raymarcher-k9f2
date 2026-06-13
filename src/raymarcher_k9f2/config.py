"""Scene configuration loader — supports YAML and TOML config files.

Config files allow defining scenes declaratively without writing Python code.

Example YAML config::

    scene:
      width: 800
      height: 600
      samples: 1
      output: my_render.png
      marcher:
        max_steps: 128
        max_bounces: 3
        enable_ao: false
        enable_soft_shadows: false
        enable_shadows: true
      camera:
        position: [7, 4, 7]
        target: [0, 1, 0]
        fov: 55
        focal_distance: 0
        aperture: 0.1
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
            checker_color: [0.3, 0.3, 0.35]
      lights:
        - type: directional
          direction: [0.5, 0.8, 0.3]
          color: [1.0, 0.95, 0.85]
          intensity: 1.0
        - type: point
          position: [2, 3, -2]
          color: [1.0, 0.8, 0.5]
          intensity: 5.0
"""

import logging
from typing import Optional

from raymarcher_k9f2.vec3 import Vec3
from raymarcher_k9f2.material import Material
from raymarcher_k9f2.primitives import (
    sdf_sphere, sdf_box, sdf_torus, sdf_cylinder, sdf_capsule,
    sdf_plane, sdf_cone,
)
from raymarcher_k9f2.scene import Scene
from raymarcher_k9f2.camera import Camera
from raymarcher_k9f2.marcher import RayMarcher
from raymarcher_k9f2.renderer import Renderer

logger = logging.getLogger(__name__)

# Mapping of primitive type names to their SDF functions
PRIMITIVE_MAP = {
    'sphere': sdf_sphere,
    'box': sdf_box,
    'torus': sdf_torus,
    'cylinder': sdf_cylinder,
    'capsule': sdf_capsule,
    'plane': sdf_plane,
    'cone': sdf_cone,
}

# Type-specific parameter names for each primitive
PRIMITIVE_PARAMS = {
    'sphere': ['center', 'radius'],
    'box': ['center', 'half_extents'],
    'torus': ['center', 'major_r', 'minor_r'],
    'cylinder': ['center', 'radius', 'half_height'],
    'capsule': ['a', 'b', 'radius'],
    'plane': ['height'],
    'cone': ['center', 'half_angle', 'height'],
}


def _parse_vec3(val) -> Vec3:
    """Parse a Vec3 from a list/tuple or Vec3."""
    if isinstance(val, Vec3):
        return val
    if isinstance(val, (list, tuple)):
        return Vec3(*val)
    raise ValueError(f"Cannot parse Vec3 from {type(val)}: {val}")


def _parse_material(d: dict) -> Material:
    """Parse a Material from a dictionary."""
    return Material.from_dict(d)


def _build_object_sdf(obj_config: dict):
    """Build an SDF function from an object config dict."""
    obj_type = obj_config.get('type', 'sphere')
    if obj_type not in PRIMITIVE_MAP:
        raise ValueError(f"Unknown primitive type: {obj_type}. Available: {list(PRIMITIVE_MAP.keys())}")

    sdf_func = PRIMITIVE_MAP[obj_type]

    # Build a closure capturing the config values
    if obj_type == 'sphere':
        center = _parse_vec3(obj_config.get('center', [0, 1, 0]))
        radius = obj_config.get('radius', 1.0)
        return lambda p: sdf_func(p, center, radius)
    elif obj_type == 'box':
        center = _parse_vec3(obj_config.get('center', [0, 0, 0]))
        half_extents = _parse_vec3(obj_config.get('half_extents', [1, 1, 1]))
        return lambda p: sdf_func(p, center, half_extents)
    elif obj_type == 'torus':
        center = _parse_vec3(obj_config.get('center', [0, 0, 0]))
        major_r = obj_config.get('major_r', 1.0)
        minor_r = obj_config.get('minor_r', 0.25)
        return lambda p: sdf_func(p, center, major_r, minor_r)
    elif obj_type == 'cylinder':
        center = _parse_vec3(obj_config.get('center', [0, 0, 0]))
        radius = obj_config.get('radius', 0.5)
        half_height = obj_config.get('half_height', 1.0)
        return lambda p: sdf_func(p, center, radius, half_height)
    elif obj_type == 'capsule':
        a = _parse_vec3(obj_config.get('a', [0, 0, 0]))
        b = _parse_vec3(obj_config.get('b', [0, 2, 0]))
        radius = obj_config.get('radius', 0.5)
        return lambda p: sdf_func(p, a, b, radius)
    elif obj_type == 'plane':
        height = obj_config.get('height', 0.0)
        return lambda p: sdf_func(p, height)
    elif obj_type == 'cone':
        center = _parse_vec3(obj_config.get('center', [0, 0, 0]))
        import math
        half_angle = obj_config.get('half_angle', 45)  # degrees
        if isinstance(half_angle, (int, float)) and half_angle > 1.0:
            half_angle = math.radians(half_angle)
        height = obj_config.get('height', 2.0)
        return lambda p: sdf_func(p, center, half_angle, height)
    else:
        raise ValueError(f"Unhandled primitive type: {obj_type}")


def load_config(path: str) -> dict:
    """Load a configuration file (YAML or TOML).

    Args:
        path: Path to the config file.

    Returns:
        Parsed configuration dictionary.
    """
    if path.endswith('.yaml') or path.endswith('.yml'):
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    elif path.endswith('.toml'):
        import toml
        with open(path, 'r') as f:
            return toml.load(f)
    else:
        # Try YAML first
        try:
            import yaml
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ValueError(f"Cannot determine format for {path}. Use .yaml or .toml extension.")


def build_scene_from_config(config: dict) -> tuple:
    """Build a Scene, Camera, Renderer, and output path from a config dict.

    Args:
        config: Configuration dictionary (from load_config).

    Returns:
        Tuple of (scene, camera, renderer, output_path).
    """
    scene_cfg = config.get('scene', config)

    # Build scene
    scene = Scene()
    scene.fog_density = scene_cfg.get('fog_density', 0.0)
    if 'fog_color' in scene_cfg:
        scene.fog_color = _parse_vec3(scene_cfg['fog_color'])

    # Add objects
    for obj_cfg in scene_cfg.get('objects', []):
        sdf_func = _build_object_sdf(obj_cfg)
        mat_cfg = obj_cfg.get('material', {})
        material = _parse_material(mat_cfg)
        scene.add_object(sdf_func, material)

    # Add lights
    for light_cfg in scene_cfg.get('lights', []):
        light_type = light_cfg.get('type', 'directional')
        if light_type == 'directional':
            direction = _parse_vec3(light_cfg.get('direction', [0.5, 0.8, 0.3]))
            color = _parse_vec3(light_cfg.get('color', [1, 1, 1])) if 'color' in light_cfg else None
            intensity = light_cfg.get('intensity', 1.0)
            scene.add_directional_light(direction, color, intensity)
        elif light_type == 'point':
            position = _parse_vec3(light_cfg.get('position', [0, 5, 0]))
            color = _parse_vec3(light_cfg.get('color', [1, 1, 1])) if 'color' in light_cfg else None
            intensity = light_cfg.get('intensity', 1.0)
            scene.add_point_light(position, color, intensity)

    # Build camera
    cam_cfg = scene_cfg.get('camera', {})
    camera = Camera(
        position=_parse_vec3(cam_cfg.get('position', [7, 4, 7])),
        target=_parse_vec3(cam_cfg.get('target', [0, 1, 0])),
        fov=cam_cfg.get('fov', 55),
        focal_distance=cam_cfg.get('focal_distance', 0.0),
        aperture=cam_cfg.get('aperture', 0.1),
    )

    # Build marcher
    marcher_cfg = scene_cfg.get('marcher', {})
    marcher = RayMarcher(
        max_steps=marcher_cfg.get('max_steps', 128),
        max_bounces=marcher_cfg.get('max_bounces', 3),
        enable_shadows=marcher_cfg.get('enable_shadows', True),
        enable_soft_shadows=marcher_cfg.get('enable_soft_shadows', False),
        enable_ao=marcher_cfg.get('enable_ao', False),
    )

    # Build renderer
    renderer = Renderer(
        width=scene_cfg.get('width', 800),
        height=scene_cfg.get('height', 600),
        samples=scene_cfg.get('samples', 1),
        marcher=marcher,
        quiet=scene_cfg.get('quiet', False),
        gamma=scene_cfg.get('gamma', 2.2),
        exposure=scene_cfg.get('exposure', 1.0),
    )

    output_path = scene_cfg.get('output', 'output.png')

    return scene, camera, renderer, output_path