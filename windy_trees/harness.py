"""Grow a source tree, simulate two-way wind, validate, render, and encode."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from physstudio.paths import ROOT
from rhizomorph.harness import encode
from .scenes import SCENES


def execute(command, env, log):
    command = list(map(str, command))
    print(shlex.join(command), flush=True)
    with log.open('w') as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=SCENES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, help='Reuse a completed Rhizomorph simulation')
    parser.add_argument('--simulation', type=Path, help='Replay a saved Windy Trees simulation')
    parser.add_argument('--stage', choices=['all', 'grow', 'simulate', 'validate', 'render', 'encode'], default='all')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--settings', type=Path, help='Wind simulation parameter overrides')
    parser.add_argument('--frames', type=int, help='Saved wind states, including frame zero')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int)
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--flow', choices=['streaks', 'pathlines', 'off'], default='streaks')
    parser.add_argument('--camera', choices=['orbit', 'static'], default='orbit')
    parser.add_argument('--blender', default=os.environ.get('BLENDER', 'blender'))
    args = parser.parse_args()
    if args.frames is not None and args.frames < 2:
        parser.error('--frames must be at least two')
    if args.height is None:
        args.height = round(args.width * 9 / 32) * 2
    if args.samples < 1 or any(n < 32 or n % 2 for n in [args.width, args.height]):
        parser.error('Use positive samples and even image dimensions of at least 32')
    if args.simulation and args.stage in ['all', 'grow', 'simulate']:
        parser.error('--simulation is for validate, render or encode stages')
    blender = shutil.which(os.path.expanduser(args.blender))
    if args.stage in ['all', 'render'] and blender is None:
        parser.error('Set BLENDER or --blender to a Blender 4.2 executable')
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = args.source.expanduser().resolve() if args.source else out / 'growth'
    sim = args.simulation.expanduser().resolve() if args.simulation else out / 'simulation'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED='1')
    for key, folder in [('WARP_CACHE_PATH', 'warp'), ('XDG_CACHE_HOME', 'blender'),
                        ('CUDA_CACHE_PATH', 'cuda'), ('OPTIX_CACHE_PATH', 'optix')]:
        env.setdefault(key, str(ROOT / '.cache' / folder))
    stages = ['grow', 'simulate', 'validate', 'render', 'encode'] if args.stage == 'all' else [args.stage]
    for stage in stages:
        if stage == 'grow':
            if not args.source:
                execute([sys.executable, '-m', 'rhizomorph.simulate', '--scene', args.scene,
                         '--output', source], env, out / 'grow.log')
            growth = json.loads((source / 'config.json').read_text())
            if growth['scene'] != args.scene or not (source / 'states' / f"{growth['steps']:05d}.npz").is_file():
                raise ValueError('The source must be a completed growth run for this scene')
            continue
        if stage == 'simulate':
            growth = json.loads((source / 'config.json').read_text())
            if growth['scene'] != args.scene:
                raise ValueError('The source tree scene must match --scene')
            command = [sys.executable, '-m', 'windy_trees.simulate', '--scene', args.scene,
                       '--source', source, '--output', sim]
            if args.settings:
                command += ['--settings', args.settings.expanduser().resolve()]
            if args.frames is not None:
                command += ['--frames', args.frames]
            execute(command, env, out / 'simulate.log')
        config = json.loads((sim / 'config.json').read_text())
        if config['scene'] != args.scene:
            raise ValueError('The simulation scene must match --scene')
        if stage == 'validate':
            execute([sys.executable, '-m', 'windy_trees.validate', '--simulations', sim,
                     '--output', out / 'validation'], env, out / 'validate.log')
        elif stage == 'render':
            execute([blender, '-b', '--factory-startup', '--threads', '8',
                     '--python-exit-code', '1', '--python', Path(__file__).with_name('render_blender.py'),
                     '--', '--input', sim, '--output', out / 'render', '--width', args.width,
                     '--height', args.height, '--samples', args.samples, '--flow', args.flow,
                     '--camera', args.camera], env, out / 'render.log')
        elif stage == 'encode':
            metadata = json.loads((out / 'render/render.json').read_text())
            total = len(list((sim / 'states').glob('*.npz')))
            if metadata['rendered'] != list(range(total)) or Path(metadata['simulation']).resolve() != sim:
                raise ValueError('Encoding requires all frames from the selected simulation')
            encode(out / 'render/frames', out / f'{args.scene}.mp4', config['fps'])
    (out / 'run.json').write_text(json.dumps(dict(scene=args.scene, source=str(source),
        simulation=str(sim), output=str(out), stage=args.stage, flow=args.flow,
        camera=args.camera), indent=2))


if __name__ == '__main__':
    main()
