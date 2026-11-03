"""Simulate cooled water and save phase, energy, and motion trajectories."""
import argparse,hashlib,json,time,shutil
from pathlib import Path
import numpy as np
from .scenes import build,PRESETS
from .solver import FreezingSystem


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',choices=PRESETS,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int);p.add_argument('--seconds',type=float);p.add_argument('--dx',type=float)
    p.add_argument('--thermal-scale',type=float);p.add_argument('--dt',type=float)
    p.add_argument('--iterations',type=int);p.add_argument('--device',default='cuda:0')
    p.add_argument('--settings',type=Path);args=p.parse_args()
    settings=json.loads(args.settings.read_text()) if args.settings else {}
    x,v,g,e,cfg=build(args.scene,args.dx,settings)
    if args.frames is not None and args.seconds is None:
        cfg['seconds']*=float(args.frames-1)/(cfg['frames']-1)
    for k in ['frames','seconds','thermal_scale']:
        if getattr(args,k) is not None:cfg[k]=getattr(args,k)
    if cfg['frames']<2 or cfg['seconds']<=0:raise ValueError('Positive duration and at least two frames required')
    args.dt=args.dt or cfg.get('mechanical_dt',1/720)
    args.iterations=args.iterations or cfg.get('projection_iterations',6)
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'states').mkdir();(args.output/'source').mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,args.output/'source'/f.name)
    cfg['source_hashes']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
    root=Path(__file__).resolve().parents[1]
    dependencies=['ice_melting/solver.py','ice_melting/thermal.py',
                  'ice_melting/scenes.py']
    for name in dependencies:
        target=args.output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,target)
    cfg['dependency_hashes']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in dependencies}
    frame_dt=cfg['seconds']/(cfg['frames']-1)
    heat_steps=int(cfg.get('thermal_substeps',1))
    if heat_steps<1:raise ValueError('thermal_substeps must be positive')
    steps=int(np.ceil(frame_dt/(heat_steps*args.dt)));dt=frame_dt/(heat_steps*steps)
    cfg.update(dt=dt,substeps=heat_steps*steps,thermal_substeps=heat_steps,iterations=args.iterations)
    (args.output/'config.json').write_text(json.dumps(cfg,indent=2))
    system=FreezingSystem(x,v,g,e,cfg,args.device);rows=[];start=time.monotonic()
    print(json.dumps(dict(scene=args.scene,particles=len(x),dt=dt,substeps=cfg['substeps'],
                         thermal_substeps=heat_steps)),flush=True)
    for frame in range(cfg['frames']):
        if frame:
            for _ in range(heat_steps):
                system.heat(frame_dt/heat_steps)
                for _ in range(steps):system.step(dt,args.iterations)
        else:
            system.grid.build(system.x,system.h)
        row=system.audit();row.update(frame=frame,elapsed=time.monotonic()-start)
        np.savez_compressed(args.output/'states'/f'{frame:05d}.npz',**system.snapshot())
        rows.append(row)
        if abs(row['relative_energy_balance'])>5.e-4:raise RuntimeError('Heat ledger error exceeded tolerance')
        if row['max_speed']>5.:raise RuntimeError('Unresolved fluid speed')
        if frame%10==0 or frame==cfg['frames']-1:
            (args.output/'audit.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
    (args.output/'complete.json').write_text(json.dumps(dict(frames=cfg['frames'],seconds=time.monotonic()-start),indent=2))


if __name__=='__main__':main()
