#!/usr/bin/env python3
"""CLI entry point for the SDF Ray Marcher.

Usage::

    raymarcher-k9f2 --scene demo --width 800 --height 600 --output output.png
    raymarcher-k9f2 --config scene.yaml --output custom.png
    raymarcher-k9f2 --scene glass --ao --soft-shadow --samples 4
    raymarcher-k9f2 --scene demo --tonemap aces --output hdr.png
    raymarcher-k9f2 --scene demo --multiprocess --output fast.png
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
from raymarcher_k9f2.tonemap import apply_tonemap, list_tonemaps

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
            pos = Vec3(7, 4, 7)
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
    elif scene_name == 'showcase':
        return Camera(Vec3(8, 4, 8), Vec3(0, 1, 0), fov=55, focal_distance=args.dof or 0, aperture=args.aperture)
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
  raymarcher-k9f2 --scene demo --tonemap aces --exposure 1.2 --output hdr.png
  raymarcher-k9f2 --scene showcase --multiprocess --output fast.png

Available scenes: demo, chess, terrain, abstract, glass, fractal, studio, showcase
Available tone mappers: aces, reinhard, filmic, clamped, linear
        """,
    )
    # Rendering
    parser.add_argument('--width', type=int, default=800, help='Image width (default: 800)')
    parser.add_argument('--height', type=int, default=600, help='Image height (default: 600)')
    parser.add_argument('--samples', type=int, default=1, help='Samples per pixel for anti-aliasing (default: 1)')
    parser.add_argument('--output', type=str, default='output.png', help='Output file path (default: output.png)')
    parser.add_argument('--scene', type=str, default='demo',
                        choices=list(SCENES.keys()), help='Built-in scene to render (default: demo)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML/TOML scene configuration file (overrides --scene)')

    # Animation
    parser.add_argument('--animate', type=int, default=1, help='Number of animation frames (default: 1)')

    # Ray marching
    parser.add_argument('--max-bounces', type=int, default=3, help='Max reflection/refraction bounces (default: 3)')
    parser.add_argument('--ao', action='store_true', help='Enable ambient occlusion')
    parser.add_argument('--soft-shadow', action='store_true', help='Enable soft shadows')
    parser.add_argument('--no-shadow', action='store_true', help='Disable shadows entirely')
    parser.add_argument('--steps', type=int, default=128, help='Max ray march steps per ray (default: 128)')

    # Depth of field
    parser.add_argument('--dof', type=float, default=0, help='Depth-of-field focal distance (0 = disabled)')
    parser.add_argument('--aperture', type=float, default=0.1, help='DOF aperture size (default: 0.1)')

    # Tone mapping
    parser.add_argument('--tonemap', type=str, default=None,
                        choices=list_tonemaps(),
                        help='Tone mapping method (default: None, use built-in gamma)')
    parser.add_argument('--exposure', type=float, default=1.0, help='Linear exposure multiplier (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=2.2, help='Gamma correction exponent (default: 2.2)')

    # Performance
    parser.add_argument('--multiprocess', action='store_true',
                        help='Use multiprocessing renderer (distributes across CPU cores)')
    parser.add_argument('--processes', type=int, default=None,
                        help='Number of processes for multiprocess rendering (default: CPU count)')

    # Output
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    parser.add_argument('--list-scenes', action='store_true', help='List available scenes and exit')
    parser.add_argument('--list-tonemaps', action='store_true', help='List available tone mappers and exit')
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
        for name, builder in sorted(SCENES.items()):
            print(f"  {name:12s} — {builder.__doc__.strip().split(chr(10))[0] if builder.__doc__ else ''}")
        return 0

    if args.list_tonemaps:
        print("Available tone mappers:")
        for name in list_tonemaps():
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

        if args.tonemap:
            img = apply_tonemap(img, method=args.tonemap, exposure=args.exposure)
            if not args.quiet:
                print(f"  Applied tone mapping: {args.tonemap}")

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

    if args.multiprocess:
        from raymarcher_k9f2.vec_renderer import MultiprocessRenderer
        renderer = MultiprocessRenderer(
            width=args.width, height=args.height, samples=args.samples,
            num_processes=args.processes,
            max_steps=args.steps, max_bounces=args.max_bounces,
            enable_shadows=not args.no_shadow, enable_ao=args.ao,
            gamma=args.gamma, exposure=args.exposure, quiet=args.quiet,
        )
        # For multiprocess, we need a simple render+save flow
        scene_obj = build_scene(0.0)
        cam = make_camera(args.scene, args)

        if not args.quiet:
            print(f"Rendering scene '{args.scene}' ({args.width}x{args.height}) with multiprocessing to {args.output}")
        img = renderer.render(scene_obj, cam)

        if args.tonemap:
            img = apply_tonemap(img, method=args.tonemap, exposure=args.exposure)
            if not args.quiet:
                print(f"  Applied tone mapping: {args.tonemap}")

        renderer.save(img, args.output)
        return 0

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
            scene_obj = build_scene(t)

            cam_base = make_camera(args.scene, args)
            if args.scene in ('demo', 'abstract', 'fractal'):
                angle = t
                r = 7 if args.scene == 'demo' else (6 if args.scene == 'abstract' else 4)
                cam_base.position = Vec3(r * math.cos(angle * 0.3), cam_base.position.y, r * math.sin(angle * 0.3))

            img = renderer.render(scene_obj, cam_base)

            if args.tonemap:
                img = apply_tonemap(img, method=args.tonemap, exposure=args.exposure)

            frame_path = f"{basename}_{frame:04d}{ext}"
            renderer.save(img, frame_path)
        if not args.quiet:
            print(f"Animation complete: {args.animate} frames")
    else:
        scene_obj = build_scene(0.0)
        cam = make_camera(args.scene, args)

        if not args.quiet:
            print(f"Rendering scene '{args.scene}' ({args.width}x{args.height}) to {args.output}")
        img = renderer.render(scene_obj, cam)

        if args.tonemap:
            img = apply_tonemap(img, method=args.tonemap, exposure=args.exposure)
            if not args.quiet:
                print(f"  Applied tone mapping: {args.tonemap}")

        renderer.save(img, args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())