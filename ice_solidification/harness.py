"""Run freezing simulation, audits, surface reconstruction, rendering, and encoding."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCENES = ('cold_finger', 'cascade', 'ice_cast')


def run(command, env, log):
    command = list(map(str, command))
    print(shlex.join(command), flush=True)
    with log.open('w') as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def encode(frames, output, fps=30):
    env = os.environ.copy()
    env.pop('LD_LIBRARY_PATH', None)
    ffmpeg = env.get('FFMPEG', 'ffmpeg')
    ffprobe = env.get('FFPROBE', 'ffprobe')
    files = sorted(frames.glob('*.png'))
    if not files or [p.stem for p in files] != [f'{i:05d}' for i in range(len(files))]:
        raise ValueError('Expected a contiguous PNG sequence starting at 00000.png')
    subprocess.run([ffmpeg, '-v', 'error', '-nostdin', '-n', '-framerate', str(fps),
                    '-i', str(frames / '%05d.png'), '-c:v', 'libx264', '-preset', 'slow',
                    '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                    str(output)], check=True, env=env)
    subprocess.run([ffmpeg, '-v', 'error', '-i', str(output), '-f', 'null', '-'],
                   check=True, env=env)
    probe = json.loads(subprocess.check_output([
        ffprobe, '-v', 'error', '-count_frames', '-show_entries',
        'stream=width,height,nb_read_frames,r_frame_rate:format=duration,size',
        '-of', 'json', str(output)], env=env))
    assert int(probe['streams'][0]['nb_read_frames']) == len(files)
    output.with_suffix('.json').write_text(json.dumps(probe, indent=2))
    return probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=SCENES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--settings', type=Path)
    parser.add_argument('--stage', choices=['all', 'validate', 'simulate', 'audit', 'surface',
                                           'render', 'encode'], default='all')
    parser.add_argument('--frames', type=int)
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--dx', type=float)
    parser.add_argument('--dt', type=float)
    parser.add_argument('--samples', type=int, default=48)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int)
    parser.add_argument('--blender', default=os.environ.get('BLENDER', 'blender'))
    args = parser.parse_args()
    if args.frames is not None and args.frames < 2:
        parser.error('--frames must be at least two')
    for name in ['seconds', 'dx', 'dt']:
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f'--{name} must be positive')
    if args.height is None:
        args.height = round(args.width * 5 / 16) * 2
    if any(size < 32 or size % 2 for size in [args.width, args.height]):
        parser.error('Image dimensions must be even and at least 32')
    if args.samples < 1:
        parser.error('--samples must be positive')
    if args.settings:
        args.settings = args.settings.expanduser().resolve()
    blender = shutil.which(os.path.expanduser(args.blender))
    if args.stage in ('all', 'render') and blender is None:
        parser.error('Set BLENDER or --blender to a Blender 4.2 executable')
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpu
    for key, name in [('WARP_CACHE_PATH', 'warp'), ('XDG_CACHE_HOME', 'blender'),
                      ('CUDA_CACHE_PATH', 'cuda'), ('OPTIX_CACHE_PATH', 'optix'),
                      ('MPLCONFIGDIR', 'matplotlib')]:
        env.setdefault(key, str(ROOT / '.cache' / name))
    sim, mesh, render = out / 'simulation', out / 'surfaces', out / 'render'
    stages = ['simulate', 'audit', 'surface', 'render', 'encode'] if args.stage == 'all' else [args.stage]

    def module(name, *options):
        run([sys.executable, '-m', 'ice_solidification.' + name, *options],
            env, out / (name + '.log'))

    for stage in stages:
        if stage == 'validate':
            module('validate', '--output', out / 'numerics.json')
        elif stage == 'simulate':
            options = ['--scene', args.scene, '--output', sim]
            for name in ['frames', 'seconds', 'dx', 'dt', 'settings']:
                value = getattr(args, name)
                if value is not None:
                    options += ['--' + name, value]
            module('simulate', *options)
        elif stage == 'audit':
            module('diagnostics', '--input', sim, '--output', out / 'audit')
        elif stage == 'surface':
            module('surface', '--input', sim, '--output', mesh)
        elif stage == 'render':
            run([blender, '-b', '--factory-startup', '--threads', '8', '--python-exit-code', '1',
                 '--python', Path(__file__).with_name('render_blender.py'), '--',
                 '--input', sim, '--meshes', mesh, '--output', render,
                 '--width', args.width, '--height', args.height, '--samples', args.samples],
                env, out / 'render.log')
        else:
            cfg = json.loads((sim / 'config.json').read_text())
            assert len(list((render / 'frames').glob('*.png'))) == cfg['frames']
            encode(render / 'frames', render / 'demo.mp4', cfg['playback_fps'])


if __name__ == '__main__':
    main()
