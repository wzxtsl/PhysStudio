"""Record reproducible magnetic rigid-body simulations for independent rendering."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .scenes import make,control
from .solver import Solver


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scene',choices=['compass_garden','magnetic_crane','magnetic_globe'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--frames',type=int)
    p.add_argument('--settings',type=Path)
    a=p.parse_args()
    if a.frames is not None and a.frames < 2:p.error('--frames must be at least two')
    cfg=make(a.scene)
    if a.settings:cfg.update(json.loads(a.settings.read_text()))
    frames=a.frames or round(cfg['duration']*cfg['fps'])+1
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'states').mkdir()
    cfg['frames']=frames;cfg['device']=a.device
    (a.output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    sources={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
    (a.output/'source_hashes.json').write_text(json.dumps(sources,indent=2)+'\n')
    solver=Solver(cfg,a.device);dt=1/cfg['fps']/cfg['substeps'];start=time.monotonic()
    assert np.min(solver.gaps(solver.x,solver.q))>0, 'Initial interpenetration'
    for frame in range(frames):
        if frame:
            for _ in range(cfg['substeps']):solver.step(dt,control)
        np.savez_compressed(a.output/'states'/f'{frame:05d}.npz',x=solver.x,q=solver.q,v=solver.v,w=solver.w,
                            moment=solver.m,time=solver.clock,permanent=solver.permanent)
        if frame%10==0 or frame==frames-1:
            print(json.dumps(dict(frame=frame,time=solver.clock,elapsed=time.monotonic()-start,
                 min_gap=float(np.min(solver.gaps(solver.x,solver.q))),last_step=solver.history[-1] if solver.history else None)),flush=True)
            (a.output/'diagnostics.json').write_text(json.dumps(solver.history,indent=2)+'\n')
    (a.output/'complete.json').write_text(json.dumps(dict(frames=frames,elapsed=time.monotonic()-start,
           accepted_steps=len(solver.history),min_gap=min([h['min_gap'] for h in solver.history],default=0)),indent=2)+'\n')

if __name__=='__main__':main()
