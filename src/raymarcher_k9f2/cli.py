#!/usr/bin/env python3
"""CLI entry point for the SDF Ray Marcher.

Usage::

    raymarcher-k9f2 --scene demo --width 800 --height 600 --output output.png
    raymarcher-k9f2 --config scene.yaml --output custom.png
    raymarcher-k9f2 --scene glass --ao --soft-shadow --samples 4
"""

import argparse
import logging
import math
import os
import sys

from raymarcher_k9f2 import (
    Vec3, Scene, Camera, Renderer, RayMarcher, Material,
    SCENES,
)
from raymarcher_k9f2.config import load_config, build_scene_from_config


def setup_logging(verbose: bool = False):
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )


def make_camera(scene_name: str, args) -> Camera:
    """Create a camera with appropriate presets for each scene."""
    if scene_name == 'demo':
        if hasattr(args, 'animate') and args.animate > 1:
            pos = Vec3(7, 4, 7)  # Will be overridden per frame
        else:
            pos = Vec3(7, 4, 7)
        return Camera(pos, Vec3(0, 1, 0), fov=55, focal_distance=args.dof, aperture=args.aperture)
    elif scene_name == 'chess':
        return Camera(Vec3(5, 5, 8), Vec3(0, 0.5, 0), fov=50, focal_distance=args.dof or 0, aperture=args.aperture)
    elif scene_name == 'terrain':
        return Camera(Vec3(8, 5, 8), Vec3(0, 1, 0), fov=60, focal_distance=args.dof or 0, aperture=args.aperture)
    elif scene_name == 'abstract':
        return Camera(Vec3(6, 4, 6), Vec3(0, 1.5, 0), fov=55, focal_distance=args.dof or 0, aperture=args.aperture)
    elif scene_name == 'glass':
        return Camera(Vec3(5, 3.5, 8), Vec3(0, 1, 0), fov=50, focal_distance=args.dof or 8, aperture=args.aperture)
    elif scene_name == 'fractal':
        return Camera(Vec3(4, 3, 4), Vec3(0, 1.5, 0), fov=50, focal_distance=args.dof or 0, aperture=args.aperture)
    elif scene_name == 'studio':
        return Camera(Vec3(6, 3, 6), Vec3(0, 1, 0), fov=50, focal_distance=args.dof or 0, aperture=args.aperture)
    else:
        return Camera(Vec3(7, 4, 7), Vec3(0, 1, 0), fov=55, focal_distance=args.dof, aperture=args.aperture)


def main():
    parser = argparse.ArgumentParser(
        description='SDF Ray Marcher k9f2 — Pure-Python 3D ray marcher with SDFs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  raymarcher-k9f2 --scene demo --width 1280 --height 960 --output render.png
  raymarcher-k9f2 --scene glass --ao --soft-shadow --samples 4
  raymarcher-k9f2 --config my_scene.yaml --output custom.png
  raymarcher-k9f2 --scene abstract --animate 8
  raymarcher-k9f2 --scene fractal --dof 10 --aperture 0.15

Available scenes: demo, chess, terrain, abstract, glass, fractal, studio
        """,
    )
    parser.add_argument('--width', type=int, default=800, help='Image width (default: 800)')
    parser.add_argument('--height', type=int, default=600, help='Image height (default: 600)')
    parser.add_argument('--samples', type=int, default=1, help='Samples per pixel for anti-aliasing (default: 1)')
    parser.add_argument('--output', type=str, default='output.png', help='Output file path (default: output.png)')
    parser.add_argument('--scene', type=str, default='demo',
                        choices=list(SCENES.keys()), help='Built-in scene to render (default: demo)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML/TOML scene configuration file (overrides --scene)')
    parser.add_argument('--animate', type=int, default=1, help='Number of animation frames (default: 1)')
    parser.add_argument('--max-bounces', type=int, default=3, help='Max reflection/refraction bounces (default: 3)')
    parser.add_argument('--ao', action='store_true', help='Enable ambient occlusion (slower)')
    parser.add_argument('--soft-shadow', action='store_true', help='Enable soft shadows (slower)')
    parser.add_argument('--no-shadow', action='store_true', help='Disable shadows entirely')
    parser.add_argument('--dof', type=float, default=0, help='Depth-of-field focal distance (0 = disabled)')
    parser.add_argument('--aperture', type=float, default=0.1, help='DOF aperture size (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=2.2, help='Gamma correction exponent (default: 2.2)')
    parser.add_argument('--exposure', type=float, default=1.0, help='Linear exposure multiplier (default: 1.0)')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    parser.add_argument('--steps', type=int, default=128, help='Max ray march steps per ray (default: 128)')
    parser.add_argument('--list-scenes', action='store_true', help='List available built-in scenes and exit')
    parser.add_argument('--version', action='store_true', help='Print version and exit')

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    if args.version:
        from raymarcher_k9f2 import __version__
        print(f"sdf-raymarcher-k9f2 version {__version__}")
        return 0

    if args.list_scenes:
        print("Available scenes:")
        for name in sorted(SCENES.keys()):
            print(f"  {name}")
        return 0

    # Config-based rendering
    if args.config:
        config = load_config(args.config)
        scene, camera, renderer, config_output = build_scene_from_config(config)
        output_path = args.output if args.output != 'output.png' else config_output

        if not args.quiet:
            print(f"Rendering config scene ({renderer.width}x{renderer.height}) to {output_path}")
        img = renderer.render(scene, camera)
        renderer.save(img, output_path)
        return 0

    # Built-in scene rendering
    build_scene = SCENES[args.scene]

    marcher = RayMarcher(
        max_steps=args.steps,
        max_bounces=args.max_bounces,
        enable_shadows=not args.no_shadow,
        enable_soft_shadows=args.soft_shadow,
        enable_ao=args.ao,
    )

    renderer = Renderer(
        width=args.width,
        height=args.height,
        samples=args.samples,
        marcher=marcher,
        quiet=args.quiet,
        gamma=args.gamma,
        exposure=args.exposure,
    )

    if args.animate > 1:
        basename = os.path.splitext(args.output)[0]
        ext = os.path.splitext(args.output)[1] or '.png'
        for frame in range(args.animate):
            t = frame / args.animate * math.pi * 2.0
            if not args.quiet:
                print(f"Frame {frame + 1}/{args.animate}")
            scene = build_scene(t)

            # Animated camera
            cam_base = make_camera(args.scene, args)
            if args.scene in ('demo', 'abstract', 'fractal'):
                angle = t
                r = 7 if args.scene == 'demo' else (6 if args.scene == 'abstract' else 4)
                cam_base.position = Vec3(r * math.cos(angle * 0.3), cam_base.position.y, r * math.sin(angle * 0.3))

            img = renderer.render(scene, cam_base)
            frame_path = f"{basename}_{frame:04d}{ext}"
            renderer.save(img, frame_path)
        if not args.quiet:
            print(f"Animation complete: {args.animate} frames")
    else:
        scene = build_scene(0.0)
        cam = make_camera(args.scene, args)

        if not args.quiet:
            print(f"Rendering scene '{args.scene}' ({args.width}x{args.height}) to {args.output}")
        img = renderer.render(scene, cam)
        renderer.save(img, args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())