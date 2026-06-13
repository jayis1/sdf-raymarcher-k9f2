"""SDF transformation utilities — translate, rotate, scale, and repeat.

These transforms modify points before passing them to SDF primitives,
enabling scene graphs with positioned, rotated, and scaled objects.

Examples::

    from raymarcher_k9f2.transforms import sdf_translate, sdf_rotate_y, sdf_scale

    # Sphere at position (3, 1, 0)
    sdf_translate(p, Vec3(3, 1, 0), lambda q: sdf_sphere(q, Vec3(0, 0, 0), 1.0))

    # Rotated torus
    sdf_rotate_y(p, math.pi / 4, lambda q: sdf_torus(q, Vec3(0, 1, 0), 1.0, 0.3))
"""

import math
from raymarcher_k9f2.vec3 import Vec3


def sdf_translate(p: Vec3, offset: Vec3, sdf_func) -> float:
    """Translate an SDF by offset.

    Shifts the point by -offset before evaluating the SDF.

    Args:
        p: Point to evaluate.
        offset: Translation offset.
        sdf_func: Callable taking a Vec3 and returning a float distance.

    Returns:
        Signed distance at the translated point.
    """
    return sdf_func(p - offset)


def sdf_scale(p: Vec3, scale: float, sdf_func) -> float:
    """Scale an SDF uniformly by a scale factor.

    Note: SDF scaling requires dividing the result by the scale factor
    to maintain correct distance properties.

    Args:
        p: Point to evaluate.
        scale: Scale factor (>0). Values <1 shrink, >1 enlarge.
        sdf_func: Callable taking a Vec3 and returning a float distance.

    Returns:
        Scaled signed distance.
    """
    return sdf_func(p * (1.0 / scale)) * scale


def sdf_rotate_y(p: Vec3, angle: float, sdf_func) -> float:
    """Rotate an SDF around the Y axis by the given angle.

    Args:
        p: Point to evaluate.
        angle: Rotation angle in radians.
        sdf_func: Callable taking a Vec3 and returning a float distance.

    Returns:
        Signed distance at the rotated point.
    """
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    qx = p.x * cos_a + p.z * sin_a
    qz = -p.x * sin_a + p.z * cos_a
    return sdf_func(Vec3(qx, p.y, qz))


def sdf_rotate_x(p: Vec3, angle: float, sdf_func) -> float:
    """Rotate an SDF around the X axis by the given angle.

    Args:
        p: Point to evaluate.
        angle: Rotation angle in radians.
        sdf_func: Callable taking a Vec3 and returning a float distance.

    Returns:
        Signed distance at the rotated point.
    """
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    qy = p.y * cos_a - p.z * sin_a
    qz = p.y * sin_a + p.z * cos_a
    return sdf_func(Vec3(p.x, qy, qz))


def sdf_rotate_z(p: Vec3, angle: float, sdf_func) -> float:
    """Rotate an SDF around the Z axis by the given angle.

    Args:
        p: Point to evaluate.
        angle: Rotation angle in radians.
        sdf_func: Callable taking a Vec3 and returning a float distance.

    Returns:
        Signed distance at the rotated point.
    """
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    qx = p.x * cos_a - p.y * sin_a
    qy = p.x * sin_a + p.y * cos_a
    return sdf_func(Vec3(qx, qy, p.z))


def sdf_repeat(p: Vec3, period: Vec3, sdf_func) -> float:
    """Repeat an SDF infinitely along all axes with the given period.

    Creates an infinite grid of the SDF shape. Useful for creating
    repeating patterns like brick walls, tile floors, etc.

    Args:
        p: Point to evaluate.
        period: Repetition period along each axis. Set an axis to
            a very large value (e.g., 1e6) to disable repetition on that axis.
        sdf_func: Callable taking a Vec3 and returning a float distance.

    Returns:
        Signed distance to the nearest repetition.
    """
    qx = p.x - round(p.x / period.x) * period.x if period.x < 1e6 else p.x
    qy = p.y - round(p.y / period.y) * period.y if period.y < 1e6 else p.y
    qz = p.z - round(p.z / period.z) * period.z if period.z < 1e6 else p.z
    return sdf_func(Vec3(qx, qy, qz))


def sdf_elongate(p: Vec3, elongation: Vec3, sdf_func) -> float:
    """Elongate an SDF along each axis.

    The elongation parameter controls how much the shape is stretched
    along each axis before the original SDF is applied.

    Args:
        p: Point to evaluate.
        elongation: Amount to elongate along each axis.
        sdf_func: Callable taking a Vec3 and returning a float distance.

    Returns:
        Signed distance of the elongated shape.
    """
    qx = p.x - clamp_val(p.x, -elongation.x, elongation.x)
    qy = p.y - clamp_val(p.y, -elongation.y, elongation.y)
    qz = p.z - clamp_val(p.z, -elongation.z, elongation.z)
    return sdf_func(Vec3(qx, qy, qz))


def clamp_val(v: float, lo: float, hi: float) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, v))