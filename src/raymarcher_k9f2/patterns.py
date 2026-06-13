"""Procedural pattern functions for materials."""

import math
from raymarcher_k9f2.vec3 import Vec3


def checkerboard(p: Vec3, scale: float, color1: Vec3, color2: Vec3) -> Vec3:
    """3D checkerboard pattern at given scale.

    Args:
        p: Point to evaluate the pattern at.
        scale: Frequency of the checkerboard pattern.
        color1: First checker color.
        color2: Second checker color.

    Returns:
        One of the two colors based on the 3D checker pattern.
    """
    ix = math.floor(p.x * scale)
    iy = math.floor(p.y * scale)
    iz = math.floor(p.z * scale)
    if (ix + iy + iz) % 2 == 0:
        return color1
    return color2


def stripes(p: Vec3, scale: float, color1: Vec3, color2: Vec3, axis: str = 'x') -> Vec3:
    """Stripe pattern along a given axis.

    Args:
        p: Point to evaluate.
        scale: Frequency of the stripes.
        color1: Color for stripe 1.
        color2: Color for stripe 2.
        axis: Axis to stripe along ('x', 'y', or 'z').

    Returns:
        One of the two colors based on the stripe pattern.
    """
    val = {'x': p.x, 'y': p.y, 'z': p.z}.get(axis, p.x)
    if math.floor(val * scale) % 2 == 0:
        return color1
    return color2


def gradient(p: Vec3, color1: Vec3, color2: Vec3, axis: str = 'y') -> Vec3:
    """Linear gradient from color1 to color2 along an axis.

    Args:
        p: Point to evaluate.
        color1: Color at the bottom (axis=0).
        color2: Color at the top (axis=1).
        axis: Axis for the gradient ('x', 'y', or 'z').

    Returns:
        Interpolated color at the point.
    """
    val = {'x': p.x, 'y': p.y, 'z': p.z}.get(axis, p.y)
    t = max(0.0, min(1.0, (val + 1.0) / 2.0))
    return color1.lerp(color2, t)