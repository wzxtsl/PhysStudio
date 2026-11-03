"""Verify and assemble frame chunks rendered with identical settings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--renders", type=Path, nargs="+", required=True)
    p.add_argument("--simulation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", default="tea_brewing")
    p.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    p.add_argument("--ffprobe", default=os.environ.get("FFPROBE", "ffprobe"))
    args = p.parse_args()
    cfg = json.loads((args.simulation / "config.json").read_text())
    settings = [
        json.loads((r / "render_config.json").read_text()) for r in args.renders
    ]
    for other in settings[1:]:
        for key in (set(other) | set(settings[0])) - {"frames"}:
            if other.get(key) != settings[0].get(key):
                raise ValueError(f"Render chunks differ: {key}")
    if Path(settings[0]["input"]).resolve() != args.simulation.resolve():
        raise ValueError("Simulation does not match renders")
    sources = {}
    for root in args.renders:
        for path in sorted((root / "frames").glob("*.png")):
            index = int(path.stem)
            if index in sources:
                if (
                    hashlib.sha256(path.read_bytes()).digest()
                    != hashlib.sha256(sources[index].read_bytes()).digest()
                ):
                    raise ValueError(f"Conflicting copies of frame {index}")
            sources[index] = path
    if set(sources) != set(range(cfg["frames"])):
        raise ValueError("Incomplete or extra frame indices")
    args.output.mkdir(parents=True, exist_ok=True)
    frames = args.output / (args.name + "_frames")
    frames.mkdir(exist_ok=False)
    for index, path in sorted(sources.items()):
        with Image.open(path) as im:
            if im.size != (settings[0]["width"], settings[0]["height"]):
                raise ValueError(f"Image dimensions differ: {path}")
            im.verify()
        dest = frames / f"{index:05d}.png"
        try:
            os.link(path, dest)
        except OSError:
            shutil.copy2(path, dest)
    video = args.output / (args.name + ".mp4")
    subprocess.run(
        [
            args.ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-n",
            "-framerate",
            str(cfg["fps"]),
            "-i",
            str(frames / "%05d.png"),
            "-frames:v",
            str(cfg["frames"]),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
    )
    probe = json.loads(
        subprocess.check_output(
            [
                args.ffprobe,
                "-v",
                "error",
                "-show_entries",
                "stream=width,height,nb_frames,r_frame_rate",
                "-show_entries",
                "format=duration,size",
                "-of",
                "json",
                str(video),
            ],
            text=True,
        )
    )
    if int(probe["streams"][0]["nb_frames"]) != cfg["frames"]:
        raise ValueError("Encoded video frame count differs")
    subprocess.run(
        [args.ffmpeg, "-v", "error", "-nostdin", "-i", str(video), "-f", "null", "-"],
        check=True,
    )
    manifest = dict(
        video=video.name,
        sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        simulation=str(args.simulation.resolve()),
        chunks=[str(r.resolve()) for r in args.renders],
        render_settings=settings[0],
        probe=probe,
    )
    (args.output / (args.name + "_manifest.json")).write_text(
        json.dumps(manifest, indent=2)
    )
    print(str(video.resolve()), flush=True)


if __name__ == "__main__":
    main()
