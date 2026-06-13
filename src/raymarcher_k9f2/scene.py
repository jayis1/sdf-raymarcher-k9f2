"""Scene definition: objects, lights, sky, fog."""

from dataclasses import dataclass, field
from typing import List, Optional, Callable

from raymarcher_k9f2.vec3 import Vec3, SKY_BLUE
from raymarcher_k9f2.material import Material


@dataclass
class SceneObject:
    """An object in the scene: SDF function + material.

    Attributes:
        sdf_func: Callable that takes a Vec3 point and returns a signed distance.
        material: Material to use when the object is hit from outside.
        interior_material: Optional material for when a ray exits this object
            (used for refractive objects with different interior/exterior shading).
    """
    sdf_func: Callable[[Vec3], float]
    material: Material
    interior_material: Optional[Material] = None


@dataclass
class HitResult:
    """Result of a ray march: distance, material, object index, and enter/exit flag.

    Attributes:
        distance: Distance from ray origin to the hit point (inf if no hit).
        material: Material at the hit point (None if no hit).
        object_index: Index of the hit object in the scene (-1 if no hit).
        entering: True if the ray is entering the object, False if exiting.
    """
    distance: float = float('inf')
    material: Optional[Material] = None
    object_index: int = -1
    entering: bool = True


@dataclass
class PointLight:
    """A point light source at a specific position.

    Attributes:
        position: World-space position of the light.
        color: RGB color of the light.
        intensity: Brightness multiplier.
    """
    position: Vec3
    color: Vec3 = field(default_factory=lambda: Vec3(1, 1, 1))
    intensity: float = 1.0


@dataclass
class DirectionalLight:
    """A directional (infinite) light source.

    Attributes:
        direction: Unit vector pointing toward the light source.
        color: RGB color of the light.
        intensity: Brightness multiplier.
    """
    direction: Vec3
    color: Vec3 = field(default_factory=lambda: Vec3(1, 1, 1))
    intensity: float = 1.0


class Scene:
    """Container for scene objects, lights, and the SDF evaluation function.

    The scene holds all objects, lights, and environment settings. The ``map()``
    method evaluates the SDF at any point to find the closest object.

    Examples::

        scene = Scene()
        scene.add_object(lambda p: sdf_sphere(p, Vec3(0, 1, 0), 1.0),
                         Material(albedo=Vec3(1, 0, 0)))
        scene.add_directional_light(Vec3(0.5, 0.8, 0.3))
    """

    def __init__(self):
        self.objects: List[SceneObject] = []
        self.directional_lights: List[DirectionalLight] = []
        self.point_lights: List[PointLight] = []
        self.sky_color: Vec3 = SKY_BLUE
        self.fog_color: Vec3 = Vec3(0.7, 0.75, 0.85)
        self.fog_density: float = 0.0
        self.time: float = 0.0
        self.background_func: Optional[Callable[[Vec3], Vec3]] = None
        self.environment_color: Vec3 = Vec3(0.4, 0.5, 0.6)

    def add_object(self, sdf_func: Callable[[Vec3], float], material: Material,
                   interior_material: Material = None) -> int:
        """Add an object to the scene. Returns the object's index."""
        idx = len(self.objects)
        self.objects.append(SceneObject(sdf_func, material, interior_material))
        return idx

    def add_directional_light(self, direction: Vec3, color: Vec3 = None, intensity: float = 1.0):
        """Add a directional light. Direction is normalized internally."""
        if color is None:
            color = Vec3(1, 1, 1)
        self.directional_lights.append(DirectionalLight(direction.normalized(), color, intensity))

    def add_point_light(self, position: Vec3, color: Vec3 = None, intensity: float = 1.0):
        """Add a point light source."""
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