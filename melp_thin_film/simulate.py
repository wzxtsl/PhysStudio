"""Record independent Warp MELP states and numerical diagnostics."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from .scenes import make
from .solver import Solver

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',default='giant_recoil');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int);p.add_argument('--settings',type=Path);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();cfg=make(a.scene)
    if a.settings:cfg.update(json.loads(a.settings.read_text()))
    cfg.setdefault('camera_frames',cfg['frames'])
    if a.frames is not None:cfg['frames']=a.frames
    if cfg['frames']<2:raise ValueError('Record at least two frames')
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'states').mkdir()
    (a.output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (a.output/'source_hashes.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},indent=2)+'\n')
    solver=Solver(cfg,a.device);start=time.monotonic();history=[]
    substeps=round(cfg['frame_dt']/cfg['dt']);assert abs(substeps*cfg['dt']-cfg['frame_dt'])<1e-12
    for frame in range(cfg['frames']):
        if frame:
            for step in range(substeps):solver.step(cfg['dt'])
        solver.refresh()
        d=solver.diagnostics();d.update(frame=frame,elapsed=time.monotonic()-start);history.append(d)
        state=solver.state();state['time']=solver.time
        np.savez_compressed(a.output/'states'/f'{frame:05d}.npz',**state)
        if frame%5==0 or frame==cfg['frames']-1:
            print(json.dumps(d),flush=True);(a.output/'diagnostics.json').write_text(json.dumps(history,indent=2)+'\n')
    (a.output/'complete.json').write_text(json.dumps(dict(frames=cfg['frames'],elapsed=time.monotonic()-start,steps=solver.step_count),indent=2)+'\n')

if __name__=='__main__':main()
