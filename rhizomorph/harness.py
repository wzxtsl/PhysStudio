"""Simulate coupled root/shoot growth, audit it, and render a compressed demo."""

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from .scenes import SCENES

ROOT = Path(__file__).resolve().parents[1]


def execute(command, env, log):
    command = list(map(str, command))
    print(shlex.join(command), flush=True)
    with log.open("w") as stream:
        subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def encode(frames, output, fps):
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    ffmpeg = env.get("FFMPEG", "ffmpeg")
    ffprobe = env.get("FFPROBE", "ffprobe")
    files = sorted(frames.glob("*.png"))
    if not files or [p.stem for p in files] != [f"{i:05d}" for i in range(len(files))]:
        raise ValueError("Expected a contiguous PNG sequence starting at 00000.png")
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-n",
            "-framerate",
            str(fps),
            "-i",
            str(frames / "%05d.png"),
            "-c:v",
            "libx264",
            "-threads",
            "8",
            "-preset",
            "slow",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(output), "-f", "null", "-"],
        env=env,
        check=True,
    )
    result = json.loads(
        subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-count_frames",
                "-show_entries",
                "stream=width,height,nb_read_frames,r_frame_rate:format=duration,size",
                "-of",
                "json",
                str(output),
            ],
            env=env,
        )
    )
    assert int(result["streams"][0]["nb_read_frames"]) == len(files)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--stage",
        choices=["all", "simulate", "audit", "render", "encode"],
        default="all",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--blender", default=os.environ.get("BLENDER", "blender"))
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.frames < 2 or args.samples < 1:
        parser.error("Use at least two frames and one rendering sample")
    if args.height is None:
        args.height = round(args.width * 9 / 16) * 2
    if any(size < 32 or size % 2 for size in (args.width, args.height)):
        parser.error("Image dimensions must be even and at least 32")
    if args.settings:
        args.settings = args.settings.expanduser().resolve()
    blender = shutil.which(os.path.expanduser(args.blender))
    if args.stage in ("all", "render") and blender is None:
        parser.error("Set BLENDER or --blender to a Blender 4.2 executable")

    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONUNBUFFERED"] = "1"
    for key, name in [
        ("WARP_CACHE_PATH", "warp"),
        ("XDG_CACHE_HOME", "blender"),
        ("CUDA_CACHE_PATH", "cuda"),
        ("OPTIX_CACHE_PATH", "optix"),
    ]:
        env.setdefault(key, str(ROOT / ".cache" / name))
    sim, render = out / "simulation", out / "render"
    stages = (
        ["simulate", "audit", "render", "encode"]
        if args.stage == "all"
        else [args.stage]
    )
    for stage in stages:
        if stage == "simulate":
            command = [
                sys.executable,
                "-m",
                "rhizomorph.simulate",
                "--scene",
                args.scene,
                "--output",
                sim,
            ]
            if args.steps is not None:
                command += ["--steps", args.steps]
            if args.settings:
                command += ["--settings", args.settings]
            execute(command, env, out / "simulate.log")
        elif stage == "audit":
            execute(
                [
                    sys.executable,
                    "-m",
                    "rhizomorph.validate",
                    "--input",
                    sim,
                    "--output",
                    out / "audit",
                ],
                env,
                out / "audit.log",
            )
        elif stage == "render":
            execute(
                [
                    blender,
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
                    sim,
                    "--output",
                    render,
                    "--frames",
                    args.frames,
                    "--width",
                    args.width,
                    "--height",
                    args.height,
                    "--samples",
                    args.samples,
                ],
                env,
                out / "render.log",
            )
        else:
            assert len(list((render / "frames").glob("*.png"))) == args.frames
            config = json.loads((sim / "config.json").read_text())
            encode(render / "frames", out / (args.scene + ".mp4"), config["fps"])


if __name__ == "__main__":
    main()
