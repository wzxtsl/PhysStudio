"""Simulate, audit, reconstruct, render and encode the three thin-film presets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from physstudio.paths import ROOT

SCENES = ('giant_recoil', 'driven_bubble', 'bubble_collision')


def execute(command, env, log):
    command = list(map(str, command))
    print(shlex.join(command), flush=True)
    with log.open('a') as stream:
        stream.write(shlex.join(command) + '\n')
        stream.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', choices=SCENES, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['all', 'simulate', 'validate', 'render', 'encode'], default='all')
    p.add_argument('--simulation', type=Path, help='Reuse a completed recording for validate, render or encode')
    p.add_argument('--gpu', default='0')
    p.add_argument('--frames', type=int, help='Saved states including frame zero; retains full physics resolution and camera timing')
    p.add_argument('--settings', type=Path, help='JSON simulation parameter overrides')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int)
    p.add_argument('--samples', type=int, default=48)
    p.add_argument('--blender', default=os.environ.get('BLENDER', 'blender'))
    p.add_argument('--start', type=int, default=0, help='First global render frame')
    p.add_argument('--end', type=int, help='Exclusive final global render frame')
    p.add_argument('--resume', action='store_true', help='Reuse meshes and PNGs with matching saved render settings')
    a = p.parse_args()
    if a.frames is not None and a.frames < 2:
        p.error('--frames must be at least two')
    if a.height is None:
        a.height = round(a.width * 9 / 32) * 2
    if a.samples < 1 or any(x < 32 or x % 2 for x in [a.width, a.height]):
        p.error('Use positive samples and even dimensions of at least 32')
    if a.start < 0 or (a.end is not None and a.end <= a.start):
        p.error('Use a nonnegative --start and --end greater than --start')
    if a.simulation and a.stage in ['all', 'simulate']:
        p.error('--simulation is for validate, render or encode')
    if a.stage != 'render' and (a.resume or a.start or a.end is not None):
        p.error('--resume, --start and --end are for the render stage')
    if a.stage not in ['all', 'simulate'] and (a.frames is not None or a.settings):
        p.error('--frames and --settings apply to simulation stages')
    blender = shutil.which(os.path.expanduser(a.blender))
    if a.stage in ['all', 'render'] and blender is None:
        p.error('Set BLENDER or --blender to a Blender 4.2 executable')
    out = a.output.expanduser().resolve()
    if a.stage == 'all' and out.exists():
        p.error('Choose a fresh --output for a complete run')
    sim = a.simulation.expanduser().resolve() if a.simulation else out / 'simulation'
    mesh, frames = out / 'mesh', out / 'render'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=a.gpu, PYTHONUNBUFFERED='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    env['PYTHONPATH'] = str(ROOT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    for key, folder in [('WARP_CACHE_PATH', 'warp'), ('XDG_CACHE_HOME', 'blender'),
                        ('CUDA_CACHE_PATH', 'cuda'), ('OPTIX_CACHE_PATH', 'optix')]:
        env.setdefault(key, str(ROOT / '.cache' / folder))
    out.mkdir(parents=True, exist_ok=True)
    stages = ['simulate', 'validate', 'render', 'encode'] if a.stage == 'all' else [a.stage]
    for stage in stages:
        if stage == 'simulate':
            cmd = [sys.executable, '-m', 'melp_thin_film.simulate', '--scene', a.scene, '--output', sim]
            if a.frames is not None:
                cmd += ['--frames', a.frames]
            if a.settings:
                cmd += ['--settings', a.settings.expanduser().resolve()]
            execute(cmd, env, out / 'simulate.log')
        config = json.loads((sim / 'config.json').read_text())
        done = json.loads((sim / 'complete.json').read_text())
        if config['scene'] != a.scene or done['frames'] != config['frames']:
            raise ValueError('Use a complete recording matching --scene')
        if stage == 'validate':
            execute([sys.executable, '-m', 'melp_thin_film.audit', '--simulation', sim,
                     '--output', out / 'audit.json'], env, out / 'validate.log')
        elif stage == 'render':
            end = config['frames'] if a.end is None else a.end
            if not 0 <= a.start < end <= config['frames']:
                raise ValueError('Render range must lie within the recorded frames')
            selected = ','.join(map(str, range(a.start, end)))
            settings = dict(simulation=str(sim), config_sha256=hashlib.sha256((sim / 'config.json').read_bytes()).hexdigest(),
                            width=a.width, height=a.height, samples=a.samples, subdivision=2,
                            material_density=a.scene == 'bubble_collision',
                            renderer_sha256=hashlib.sha256(Path(__file__).with_name('render_blender.py').read_bytes()).hexdigest())
            settings_path = out / 'render_settings.json'
            if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
                raise ValueError('Changed rendering settings require a fresh --output')
            if (any(mesh.glob('*.npz')) or any(frames.glob('*.png'))) and not a.resume:
                raise ValueError('Use --resume for matching renders or choose a fresh --output')
            if a.resume:
                from PIL import Image
                for f in frames.glob('*.png'):
                    with Image.open(f) as image:
                        if image.size != (a.width, a.height):
                            raise ValueError('Existing PNG dimensions differ from the requested render')
                        image.verify()
            settings_path.write_text(json.dumps(settings, indent=2) + '\n')
            cmd = [sys.executable, '-m', 'melp_thin_film.prepare_render', '--simulation', sim,
                   '--output', mesh, '--indices', selected, '--subdivision', 2]
            if settings['material_density']:
                cmd += ['--material-density']
            if a.resume:
                cmd += ['--resume']
            execute(cmd, env, out / 'reconstruct.log')
            execute([sys.executable, '-m', 'melp_thin_film.audit_mesh', '--simulation', sim,
                     '--mesh', mesh, '--output', out / 'mesh_audit.json'], env, out / 'validate_mesh.log')
            cmd = [blender, '-b', '--factory-startup', '--threads', 4, '--python-exit-code', 1,
                   '--python', Path(__file__).with_name('render_blender.py'), '--',
                   '--input', mesh, '--output', frames, '--indices', selected,
                   '--width', a.width, '--height', a.height, '--samples', a.samples]
            if a.resume:
                cmd += ['--resume']
            execute(cmd, env, out / 'render.log')
        elif stage == 'encode':
            settings = json.loads((out / 'render_settings.json').read_text())
            if settings['simulation'] != str(sim) or settings['config_sha256'] != hashlib.sha256((sim / 'config.json').read_bytes()).hexdigest():
                raise ValueError('Encoding requires renders from the selected recording')
            execute([sys.executable, '-m', 'melp_thin_film.finish_media', '--simulation', sim,
                     '--frames', frames, '--output', out / f'{a.scene}.mp4'], env, out / 'encode.log')
    (out / 'run.json').write_text(json.dumps(dict(scene=a.scene, simulation=str(sim),
        output=str(out), stage=a.stage, frames=config['frames']), indent=2) + '\n')


if __name__ == '__main__':
    main()
