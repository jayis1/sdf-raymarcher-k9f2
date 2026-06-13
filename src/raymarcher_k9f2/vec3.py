"""Vector math utilities for 3D ray marching."""

import math
from typing import Optional


class Vec3:
    """Minimal 3D vector class for ray marching math.

    Supports arithmetic operations, dot/cross products, reflection,
    refraction, lerp, clamp, and component-wise power.

    Examples::

        v = Vec3(1, 2, 3)
        v.length()  # 3.7416...
        v.normalized()  # Vec3(0.2672, 0.5345, 0.8018)
        v.dot(Vec3(4, 5, 6))  # 32.0
        v.cross(Vec3(0, 1, 0))  # Vec3(-3.0, 0.0, 1.0)
    """

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec3):
            return NotImplemented
        return abs(self.x - other.x) < 1e-10 and abs(self.y - other.y) < 1e-10 and abs(self.z - other.z) < 1e-10

    def dot(self, other: 'Vec3') -> float:
        """Compute the dot product with another Vec3."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vec3') -> 'Vec3':
        """Compute the cross product with another Vec3."""
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        """Compute the Euclidean length (magnitude)."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def length_sq(self) -> float:
        """Squared length — avoids sqrt when comparing distances."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalized(self) -> 'Vec3':
        """Return a unit-length copy of this vector. Returns zero vector if length is near-zero."""
        l = self.length()
        if l < 1e-12:
            return Vec3(0, 0, 0)
        inv = 1.0 / l
        return Vec3(self.x * inv, self.y * inv, self.z * inv)

    def reflect(self, normal: 'Vec3') -> 'Vec3':
        """Reflect this vector off a surface with given normal."""
        return self - normal * (2.0 * self.dot(normal))

    def refract(self, normal: 'Vec3', eta: float) -> Optional['Vec3']:
        """Refract this vector through a surface. Returns None for total internal reflection.

        Args:
            normal: Surface normal (must point toward the side the ray comes from).
            eta: Ratio n1/n2 per Snell's law (e.g. 1/1.5 for air→glass).
        """
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
        """Linear interpolation between this vector and another."""
        return self * (1.0 - t) + other * t

    def clamp(self, lo: float = 0.0, hi: float = 1.0) -> 'Vec3':
        """Clamp each component to [lo, hi]."""
        return Vec3(
            max(lo, min(hi, self.x)),
            max(lo, min(hi, self.y)),
            max(lo, min(hi, self.z)),
        )

    def pow(self, exponent: float) -> 'Vec3':
        """Component-wise power (clamps negatives to zero)."""
        return Vec3(
            pow(max(self.x, 0.0), exponent),
            pow(max(self.y, 0.0), exponent),
            pow(max(self.z, 0.0), exponent),
        )

    def to_tuple(self) -> tuple:
        """Convert to (x, y, z) tuple."""
        return (self.x, self.y, self.z)

    @classmethod
    def from_tuple(cls, t: tuple) -> 'Vec3':
        """Create a Vec3 from a (x, y, z) tuple or list."""
        return cls(t[0], t[1], t[2])


# ─── Color Constants ────────────────────────────────────────────────────────

BLACK = Vec3(0, 0, 0)
WHITE = Vec3(1, 1, 1)
SKY_BLUE = Vec3(0.5, 0.7, 1.0)