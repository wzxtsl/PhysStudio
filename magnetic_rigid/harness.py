"""Simulate, audit, render and encode the magnetic rigid-body presets."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from physstudio.paths import ROOT

SCENES = ('compass_garden', 'magnetic_crane', 'magnetic_globe')


def execute(command, env, log):
    command = list(map(str, command))
    print(shlex.join(command), flush=True)
    with log.open('a') as stream:
        stream.write(shlex.join(command) + '\n')
        stream.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=SCENES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['all', 'simulate', 'validate', 'render', 'encode'], default='all')
    parser.add_argument('--simulation', type=Path, help='Reuse a completed recording for validation, rendering or encoding')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--frames', type=int, help='Saved states, including frame zero; retains the normal time step')
    parser.add_argument('--settings', type=Path, help='JSON simulation parameter overrides')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--blender', default=os.environ.get('BLENDER', 'blender'))
    parser.add_argument('--start', type=int, default=0, help='First global render frame')
    parser.add_argument('--end', type=int, help='Exclusive final global render frame')
    parser.add_argument('--resume', action='store_true', help='Keep existing PNGs with matching render settings')
    args = parser.parse_args()
    if args.frames is not None and args.frames < 2:
        parser.error('--frames must be at least two')
    if args.height is None:
        args.height = round(args.width * 9 / 32) * 2
    if args.samples < 1 or any(n < 32 or n % 2 for n in [args.width, args.height]):
        parser.error('Use positive samples and even image dimensions of at least 32')
    if args.start < 0 or (args.end is not None and args.end <= args.start):
        parser.error('Use a nonnegative --start and --end greater than --start')
    if args.simulation and args.stage in ['all', 'simulate']:
        parser.error('--simulation is for validate, render or encode stages')
    if args.stage != 'render' and (args.resume or args.start or args.end is not None):
        parser.error('--resume, --start and --end are for the render stage')
    blender = shutil.which(os.path.expanduser(args.blender))
    if args.stage in ['all', 'render'] and blender is None:
        parser.error('Set BLENDER or --blender to a Blender 4.2 executable')
    out = args.output.expanduser().resolve()
    if args.stage == 'all' and out.exists():
        parser.error('Choose a fresh --output directory for a complete run')
    sim = args.simulation.expanduser().resolve() if args.simulation else out / 'simulation'
    render = out / 'render'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    env['PYTHONPATH'] = str(ROOT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    for key, folder in [('WARP_CACHE_PATH', 'warp'), ('XDG_CACHE_HOME', 'blender'),
                        ('CUDA_CACHE_PATH', 'cuda'), ('OPTIX_CACHE_PATH', 'optix')]:
        env.setdefault(key, str(ROOT / '.cache' / folder))
    out.mkdir(parents=True, exist_ok=True)
    stages = ['simulate', 'validate', 'render', 'encode'] if args.stage == 'all' else [args.stage]
    for stage in stages:
        if stage == 'simulate':
            command = [sys.executable, '-m', 'magnetic_rigid.simulate', '--scene', args.scene, '--output', sim]
            if args.frames is not None:
                command += ['--frames', args.frames]
            if args.settings:
                command += ['--settings', args.settings.expanduser().resolve()]
            execute(command, env, out / 'simulate.log')
        config = json.loads((sim / 'config.json').read_text())
        complete = json.loads((sim / 'complete.json').read_text())
        if config['scene'] != args.scene or complete['frames'] != config['frames']:
            raise ValueError('Use a complete recording matching --scene')
        if stage == 'validate':
            execute([sys.executable, '-m', 'magnetic_rigid.audit', sim,
                     '--output', out / 'audit.json'], env, out / 'validate.log')
        elif stage == 'render':
            if args.start >= config['frames'] or (args.end is not None and args.end > config['frames']):
                raise ValueError('Render range must lie within the recorded frames')
            if any(render.glob('*.png')) and not args.resume:
                raise ValueError('Use --resume to continue matching renders or choose a fresh --output')
            for metadata in render.glob('render_settings_*.json'):
                settings = json.loads(metadata.read_text())
                if (Path(settings['simulation']).resolve() != sim or
                        any(settings[k] != getattr(args, k) for k in ['width', 'height', 'samples'])):
                    raise ValueError('Changed render settings require a fresh --output')
            command = [blender, '-b', '--factory-startup', '--threads', '4', '--python-exit-code', '1',
                       '--python', Path(__file__).with_name('render_blender.py'), '--',
                       '--simulation', sim, '--output', render, '--width', args.width,
                       '--height', args.height, '--samples', args.samples, '--start', args.start]
            if args.end is not None:
                command += ['--end', args.end]
            if args.resume:
                command += ['--resume']
            execute(command, env, out / 'render.log')
        elif stage == 'encode':
            metadata = list(render.glob('render_settings_*.json'))
            if not metadata or any(Path(json.loads(p.read_text())['simulation']).resolve() != sim for p in metadata):
                raise ValueError('Encoding requires renders from the selected simulation')
            execute([sys.executable, '-m', 'magnetic_rigid.finish_media', '--simulation', sim,
                     '--frames', render, '--output', out / f'{args.scene}.mp4'], env, out / 'encode.log')
    (out / 'run.json').write_text(json.dumps(dict(scene=args.scene, simulation=str(sim),
        output=str(out), stage=args.stage, frames=config['frames']), indent=2) + '\n')


if __name__ == '__main__':
    main()
