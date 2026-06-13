"""CSG (Constructive Solid Geometry) operations for combining SDFs."""


def sdf_union(d1: float, d2: float) -> float:
    """Boolean union (min) of two SDF values."""
    return min(d1, d2)


def sdf_smooth_union(d1: float, d2: float, k: float = 0.5) -> float:
    """Smooth boolean union with blending factor k.

    Args:
        d1: First SDF value.
        d2: Second SDF value.
        k: Blending radius — larger values produce smoother unions.
    """
    h = max(k - abs(d1 - d2), 0.0) / k
    return min(d1, d2) - h * h * k * 0.25


def sdf_smooth_subtraction(d1: float, d2: float, k: float = 0.5) -> float:
    """Smooth subtraction: d1 minus d2, with smooth blending factor k.

    Args:
        d1: First SDF value (base shape).
        d2: Second SDF value (subtracted shape).
        k: Blending radius.
    """
    h = max(k - abs(-d2 - d1), 0.0) / k
    return max(-d2, d1) + h * h * k * 0.25


def sdf_subtraction(d1: float, d2: float) -> float:
    """Subtract d2 from d1 (d1 - d2). Result is inside where d1 is inside and d2 is outside."""
    return max(-d2, d1)


def sdf_intersection(d1: float, d2: float) -> float:
    """Boolean intersection of two SDF values. Result is inside where both are inside."""
    return max(d1, d2)


def sdf_smooth_intersection(d1: float, d2: float, k: float = 0.5) -> float:
    """Smooth boolean intersection with blending factor k.

    Args:
        d1: First SDF value.
        d2: Second SDF value.
        k: Blending radius.
    """
    h = max(k - abs(d1 - d2), 0.0) / k
    return max(d1, d2) + h * h * k * 0.25