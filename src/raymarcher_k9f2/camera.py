"""Perspective camera with look-at targeting and optional depth-of-field."""

import math
from typing import Tuple

from raymarcher_k9f2.vec3 import Vec3


class Camera:
    """Perspective camera with look-at targeting and optional depth-of-field.

    Args:
        position: Camera position in world space.
        target: Look-at target point.
        fov: Vertical field of view in degrees.
        up: Up direction vector (default: Y-up).
        focal_distance: Distance to the focal plane (0 = no DOF).
        aperture: Size of the camera aperture for DOF effects.
    """

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

        Returns:
            Tuple of (origin, direction) for the ray.
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

    def to_dict(self) -> dict:
        """Serialize camera settings to a dictionary."""
        return {
            'position': [self.position.x, self.position.y, self.position.z],
            'target': [self.target.x, self.target.y, self.target.z],
            'fov': self.fov,
            'up': [self.up.x, self.up.y, self.up.z],
            'focal_distance': self.focal_distance,
            'aperture': self.aperture,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Camera':
        """Create a Camera from a dictionary."""
        pos = Vec3(*d['position'])
        tgt = Vec3(*d['target'])
        up = Vec3(*d['up']) if 'up' in d else None
        return cls(
            position=pos,
            target=tgt,
            fov=d.get('fov', 60.0),
            up=up,
            focal_distance=d.get('focal_distance', 0.0),
            aperture=d.get('aperture', 0.1),
        )