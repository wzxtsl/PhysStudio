"""Simulate, audit, reconstruct and render the Warp multi-fluid demonstrations."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["all", "validate", "simulate", "audit", "mesh", "render", "encode"],
    )
    parser.add_argument(
        "--scene", choices=["tea", "separation", "collision"], default="tea"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dx", type=float)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--resume-render", action="store_true")
    parser.add_argument("--tea-absorption", type=float)
    parser.add_argument("--tea-color", type=float, nargs=3)
    parser.add_argument("--surface-scale", type=float)
    parser.add_argument("--kernel-width", type=float)
    parser.add_argument("--smooth-passes", type=int)
    parser.add_argument("--blender", default=os.environ.get("BLENDER", "blender"))
    args = parser.parse_args()
    if args.frames is not None and args.frames < 2:
        parser.error("--frames must be at least two")
    if args.dx is not None and args.dx <= 0:
        parser.error("--dx must be positive")
    if args.height is None:
        args.height = round(args.width * 5 / 16) * 2
    if any(size < 32 or size % 2 for size in [args.width, args.height]):
        parser.error("Image dimensions must be even and at least 32")
    if args.samples < 1:
        parser.error("--samples must be positive")
    out = args.output.expanduser().resolve()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    for name, folder in [
        ("WARP_CACHE_PATH", "warp"),
        ("XDG_CACHE_HOME", "blender"),
        ("CUDA_CACHE_PATH", "cuda"),
        ("OPTIX_CACHE_PATH", "optix"),
    ]:
        env.setdefault(name, str(ROOT / ".cache" / folder))
    stages = (
        ["validate", "simulate", "audit", "mesh", "render", "encode"]
        if args.stage == "all"
        else [args.stage]
    )

    def call(command):
        command = list(map(str, command))
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    def module(name, *options):
        call([sys.executable, "-m", "implicit_mixture." + name, *options])

    for stage in stages:
        if stage == "validate":
            module("validate", "--output", out / "numerics.json")
        elif stage == "simulate":
            options = ["--scene", args.scene, "--output", out / "simulation"]
            for name in ["dx", "frames", "config"]:
                value = getattr(args, name)
                if value is not None:
                    options += ["--" + name, value]
            module("simulate", *options)
        elif stage == "audit":
            module("audit", out / "simulation")
        elif stage == "mesh":
            options = ["--input", out / "simulation", "--output", out / "meshes"]
            for key, flag in [
                ("surface_scale", "scale"),
                ("kernel_width", "kernel-width"),
                ("smooth_passes", "smooth-passes"),
            ]:
                value = getattr(args, key)
                if value is not None:
                    options += ["--" + flag, value]
            module("surface", *options)
        elif stage == "render":
            command = [
                args.blender,
                "-b",
                "--factory-startup",
                "--threads",
                "8",
                "--python-exit-code",
                "1",
                "--python",
                Path(__file__).with_name("render_blender.py"),
                "--",
                "--input",
                out / "simulation",
                "--meshes",
                out / "meshes",
                "--output",
                out / "render",
                "--width",
                args.width,
                "--height",
                args.height,
                "--samples",
                args.samples,
            ]
            if args.resume_render:
                command += ["--resume"]
            if args.tea_absorption is not None:
                command += ["--tea-absorption", args.tea_absorption]
            if args.tea_color is not None:
                command += ["--tea-color", *args.tea_color]
            call(command)
        else:
            module(
                "assemble",
                "--renders",
                out / "render",
                "--simulation",
                out / "simulation",
                "--output",
                out / "render",
                "--name",
                "demo",
            )


if __name__ == "__main__":
    main()
