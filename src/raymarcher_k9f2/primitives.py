"""SDF primitive functions for common 3D shapes."""

import math
from raymarcher_k9f2.vec3 import Vec3


def sdf_sphere(p: Vec3, center: Vec3, radius: float) -> float:
    """Signed distance to a sphere.

    Args:
        p: Point to evaluate.
        center: Center of the sphere.
        radius: Radius of the sphere.

    Returns:
        Negative inside, zero on surface, positive outside.
    """
    return (p - center).length() - radius


def sdf_box(p: Vec3, center: Vec3, half_extents: Vec3) -> float:
    """Signed distance to an axis-aligned box.

    Uses the standard SDF formulation:
    - Outside: Euclidean distance to nearest face/edge/corner
    - Inside: negative of the minimum distance to a face (closest face depth)

    Args:
        p: Point to evaluate.
        center: Center of the box.
        half_extents: Half-size along each axis.
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
    """Signed distance to a torus in the XZ plane.

    Args:
        p: Point to evaluate.
        center: Center of the torus.
        major_r: Distance from center to tube center.
        minor_r: Radius of the tube.
    """
    q = p - center
    xz_dist = math.sqrt(q.x * q.x + q.z * q.z) - major_r
    return math.sqrt(xz_dist * xz_dist + q.y * q.y) - minor_r


def sdf_cylinder(p: Vec3, center: Vec3, radius: float, half_height: float) -> float:
    """Signed distance to a vertical cylinder.

    Args:
        p: Point to evaluate.
        center: Center of the cylinder.
        radius: Radius of the cylinder.
        half_height: Half-height of the cylinder.
    """
    q = p - center
    dist_xz = math.sqrt(q.x * q.x + q.z * q.z) - radius
    dist_y = abs(q.y) - half_height
    outside = math.sqrt(max(dist_xz, 0.0) ** 2 + max(dist_y, 0.0) ** 2)
    inside = min(max(dist_xz, dist_y), 0.0)
    return outside + inside


def sdf_capsule(p: Vec3, a: Vec3, b: Vec3, radius: float) -> float:
    """Signed distance to a capsule (line-segment swept sphere).

    Args:
        p: Point to evaluate.
        a: Start point of the line segment.
        b: End point of the line segment.
        radius: Radius of the capsule.
    """
    ab = b - a
    ap = p - a
    t = ap.dot(ab) / max(ab.dot(ab), 1e-12)
    t = max(0.0, min(1.0, t))
    closest = a + ab * t
    return (p - closest).length() - radius


def sdf_plane(p: Vec3, height: float = 0.0) -> float:
    """Signed distance to a horizontal plane at given height.

    Args:
        p: Point to evaluate.
        height: Y-coordinate of the plane.
    """
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
    sin_a = math.sin(half_angle)
    cos_a = math.cos(half_angle)
    d_xz = math.sqrt(q.x * q.x + q.z * q.z)
    # Cone surface SDF: cos_a * d_xz + sin_a * (q.y - height)
    cone_dist = cos_a * d_xz + sin_a * (q.y - height)
    # Cap at base (y = 0 relative to center)
    base_dist = -q.y

    if q.y < 0:
        # Below base: distance is Euclidean
        return math.sqrt(d_xz * d_xz + q.y * q.y)

    return max(cone_dist, base_dist)


def sdf_hex_prism(p: Vec3, center: Vec3, radius: float, half_height: float) -> float:
    """Signed distance to a hexagonal prism (vertical, in XZ plane).

    Uses the intersection of six half-planes for the hexagonal cross-section,
    extruded along Y.

    Args:
        p: Point to evaluate.
        center: Center of the prism.
        radius: Circumscribed radius of the hexagon.
        half_height: Half-height of the prism.
    """
    q = p - center
    k = math.sqrt(3.0) * 0.5
    qx_abs = abs(q.x)
    qz_abs = abs(q.z)
    d2 = qx_abs - radius
    d3 = qz_abs - radius * k
    d4 = qx_abs * 0.5 + qz_abs * k - radius
    d_hex = max(d2, max(d3, d4))
    d_y = abs(q.y) - half_height
    outside = math.sqrt(max(d_hex, 0.0) ** 2 + max(d_y, 0.0) ** 2)
    inside = min(max(d_hex, d_y), 0.0)
    return outside + inside


def sdf_mandelbulb(p: Vec3, center: Vec3, scale: float = 1.0, power: float = 8.0,
                   iterations: int = 8) -> float:
    """Approximate distance estimator for the Mandelbulb fractal.

    This is a ray-marchable distance estimator, not an exact SDF —
    we use a normalized orbit trap method.

    Args:
        p: Point to evaluate.
        center: Center of the Mandelbulb.
        scale: Scale factor for the fractal.
        power: Power parameter (classic Mandelbulb uses 8).
        iterations: Number of iterations for the distance estimator.
    """
    z = p - center
    z = z * scale
    dr = 1.0
    r = 0.0

    for _ in range(iterations):
        r = z.length()
        if r > 2.0:
            break
        theta = math.acos(max(-1.0, min(1.0, z.z / max(r, 1e-12))))
        phi = math.atan2(z.y, z.x)
        dr = pow(r, power - 1.0) * power * dr + 1.0

        zr = pow(r, power)
        theta *= power
        phi *= power
        z = Vec3(
            zr * math.sin(theta) * math.cos(phi),
            zr * math.sin(theta) * math.sin(phi),
            zr * math.cos(theta),
        ) + (p - center) * scale

    return 0.5 * math.log(max(r, 1e-12)) * r / max(dr, 1.0) / scale