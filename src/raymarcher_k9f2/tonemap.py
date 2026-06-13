"""Tone mapping operators for HDR-to-LDR conversion.

Provides industry-standard tone mappers that can be applied to the
rendered image for better dynamic range handling.

Usage::

    from raymarcher_k9f2.tonemap import apply_tonemap

    img = renderer.render(scene, camera)
    img_ldr = apply_tonemap(img, method='aces')

Supported methods:
    - 'linear': No tone mapping (pass-through)
    - 'reinhard': Classic Reinhard tone mapping
    - 'aces': ACES filmic tone mapping (industry standard)
    - 'filmic': Uncharted 2 filmic curve
    - 'clamped': Simple clamp to [0, 1]
"""

import numpy as np
from typing import Optional


def _reinhard(x: np.ndarray) -> np.ndarray:
    """Reinhard tone mapping: x / (1 + x).

    Simple and effective for most scenes. Tends to wash out very bright areas.
    """
    return x / (1.0 + x)


def _aces(x: np.ndarray) -> np.ndarray:
    """ACES filmic tone mapping (approximation).

    The industry standard for film and games. Preserves highlights
    better than Reinhard and gives a cinematic look.

    Based on Stephen Hill's fit of Krzysztof Narkowicz's ACES curve.
    """
    # Narkowicz ACES approximation
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def _filmic_uncharted2(x: np.ndarray) -> np.ndarray:
    """Uncharted 2 filmic tone mapping.

    A popular curve from the Uncharted 2 presentation by John Hable.
    Gives a nice cinematic look with good highlight handling.
    """
    A = 0.22  # Shoulder Strength
    B = 0.30  # Linear Strength
    C = 0.10  # Linear Angle
    D = 0.20  # Toe Strength
    E = 0.01  # Toe Numerator
    F = 0.30  # Toe Denominator

    def curve(v):
        return ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F

    white_point = 11.4
    return curve(x) / curve(np.full_like(x, white_point))


def _clamped(x: np.ndarray) -> np.ndarray:
    """Simple clamping to [0, 1]. Destructive for HDR content."""
    return np.clip(x, 0.0, 1.0)


_TONEMAP_FUNCS = {
    'linear': lambda x: x,
    'reinhard': _reinhard,
    'aces': _aces,
    'filmic': _filmic_uncharted2,
    'clamped': _clamped,
}


def apply_tonemap(img: np.ndarray, method: str = 'aces',
                  exposure: float = 1.0) -> np.ndarray:
    """Apply tone mapping to an HDR image.

    Args:
        img: (H, W, 3) float array, values can exceed [0, 1] for HDR.
        method: Tone mapping method: 'aces', 'reinhard', 'filmic',
                'clamped', or 'linear'.
        exposure: Pre-tonemap exposure multiplier.

    Returns:
        (H, W, 3) float array with values in [0, 1].

    Raises:
        ValueError: If method is not recognized.
    """
    if method not in _TONEMAP_FUNCS:
        raise ValueError(
            f"Unknown tone mapping method: {method!r}. "
            f"Available: {list(_TONEMAP_FUNCS.keys())}"
        )

    # Apply exposure
    exposed = img * exposure

    # Apply tone mapping
    result = _TONEMAP_FUNCS[method](exposed)

    # Gamma correction (sRGB curve)
    result = np.clip(result, 0.0, None)
    result = np.power(result, 1.0 / 2.2)

    return np.clip(result, 0.0, 1.0)


def list_tonemaps() -> list:
    """Return a list of available tone mapping method names."""
    return list(_TONEMAP_FUNCS.keys())