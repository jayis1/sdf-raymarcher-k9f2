"""Renderer: turns scenes into images using the ray marcher."""

import time
import logging
from typing import Optional

import numpy as np
from PIL import Image

from raymarcher_k9f2.vec3 import Vec3, BLACK
from raymarcher_k9f2.marcher import RayMarcher
from raymarcher_k9f2.scene import Scene
from raymarcher_k9f2.camera import Camera

logger = logging.getLogger(__name__)


class Renderer:
    """Turns scenes into images using the ray marcher.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        samples: Samples per pixel for anti-aliasing.
        marcher: RayMarcher instance (created with defaults if None).
        quiet: Suppress progress output.
        gamma: Gamma correction exponent (default 2.2).
        exposure: Linear exposure multiplier (default 1.0).
    """

    def __init__(self, width: int = 800, height: int = 600, samples: int = 1,
                 marcher: Optional[RayMarcher] = None, quiet: bool = False,
                 gamma: float = 2.2, exposure: float = 1.0):
        self.width = width
        self.height = height
        self.samples = samples
        self.marcher = marcher or RayMarcher()
        self.quiet = quiet
        self.gamma = gamma
        self.exposure = exposure

    def render(self, scene: Scene, camera: Camera) -> np.ndarray:
        """Render the scene, returning an (H, W, 3) float array.

        Args:
            scene: The scene to render.
            camera: The camera to render from.

        Returns:
            numpy array of shape (H, W, 3) with float values in [0, 1].
        """
        img = np.zeros((self.height, self.width, 3), dtype=np.float64)
        aspect = self.width / self.height

        total_pixels = self.width * self.height
        last_pct = -1
        t_start = time.time()

        for y in range(self.height):
            pct = int((y * self.width) / total_pixels * 100)
            if pct != last_pct and pct % 10 == 0 and not self.quiet:
                elapsed = time.time() - t_start
                remaining = (elapsed / max(pct, 1)) * (100 - pct) if pct > 0 else 0
                last_pct = pct
                logger.info(f"  Rendering: {pct}% ({elapsed:.1f}s, ~{remaining:.0f}s remaining)")
                print(f"  Rendering: {pct}% ({elapsed:.1f}s, ~{remaining:.0f}s remaining)")

            for x in range(self.width):
                color = BLACK
                for s in range(self.samples):
                    # Jittered sampling with golden ratio for better distribution
                    if self.samples > 1:
                        jx = (x + (s * 0.618033988749895 % 1.0)) / self.width
                        jy = (y + (s * 0.414213562373095 % 1.0)) / self.height
                    else:
                        jx = (x + 0.5) / self.width
                        jy = (y + 0.5) / self.height

                    # Normalized coordinates in [-0.5, 0.5]
                    u = jx - 0.5
                    v = 0.5 - jy  # Flip Y so up is positive

                    origin, direction = camera.get_ray(u, v, aspect, s)
                    sample_color = self.marcher.trace(origin, direction, scene)
                    color = color + sample_color

                color = color * (1.0 / self.samples)
                # Apply exposure and gamma correction
                color = (color * self.exposure).pow(1.0 / self.gamma)
                img[y, x, 0] = color.x
                img[y, x, 1] = color.y
                img[y, x, 2] = color.z

        elapsed = time.time() - t_start
        if not self.quiet:
            logger.info(f"  Rendering: 100% ({elapsed:.1f}s)")
            print(f"  Rendering: 100% ({elapsed:.1f}s)")
        return img

    def save(self, img: np.ndarray, path: str):
        """Save float image array as 8-bit PNG.

        Args:
            img: (H, W, 3) float array from render().
            path: Output file path.
        """
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, 'RGB')
        pil_img.save(path)
        if not self.quiet:
            logger.info(f"  Saved: {path} ({img.shape[1]}x{img.shape[0]})")
            print(f"  Saved: {path} ({img.shape[1]}x{img.shape[0]})")

    def render_to_pil(self, scene: Scene, camera: Camera) -> Image.Image:
        """Render the scene and return a PIL Image directly (no file I/O).

        Args:
            scene: The scene to render.
            camera: The camera to render from.

        Returns:
            PIL Image object.
        """
        img = self.render(scene, camera)
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_uint8, 'RGB')