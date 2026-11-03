"""Audit complete recordings and quantify each advertised demo interaction."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .solver import Solver,rotate


def audit(path, require_full_demo=False):
    cfg=json.loads((path/'config.json').read_text());complete=json.loads((path/'complete.json').read_text())
    files=sorted((path/'states').glob('*.npz'));assert len(files)==complete['frames']==cfg['frames']
    assert [int(f.stem) for f in files]==list(range(len(files)))
    states=[dict(np.load(f)) for f in files];x=np.stack([s['x'] for s in states]);q=np.stack([s['q'] for s in states])
    assert all(np.isfinite(s[k]).all() for s in states for k in ['x','q','v','w','moment'])
    assert np.max(np.abs(np.linalg.norm(q,axis=-1)-1))<1.e-10
    times=np.array([float(s['time']) for s in states]);assert np.max(np.abs(times-np.arange(len(files))/cfg['fps']))<1.e-8
    full_demo=bool(times[-1]>=cfg['duration']-1.e-8)
    if require_full_demo:assert full_demo, 'A full-duration recording is required'
    history=json.loads((path/'diagnostics.json').read_text())
    assert min(h['min_gap'] for h in history)>0
    assert max(h['residual'] for h in history)<cfg['acceleration_tolerance']
    # Independently recompute every saved frame's sphere gaps from configuration.
    solver=Solver(cfg,'cpu');gaps=[float(solver.gaps(a,b).min()) for a,b in zip(x,q)];assert min(gaps)>0
    report=dict(full_demo_checks=full_demo,scene=cfg['scene'],frames=len(files),duration=float(times[-1]),bodies=len(cfg['bodies'])-1,
      magnetic_samples=len(solver.samples),min_saved_gap=min(gaps),min_substep_gap=min(h['min_gap'] for h in history),
      max_residual=max(h['residual'] for h in history),accepted_steps=len(history),
      max_retry_depth=max(h['retry_depth'] for h in history),max_newton=max(h['newton'] for h in history),
      max_kappa=max(h['kappa'] for h in history),simulation_seconds=complete['elapsed'],
      state_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    if cfg['scene']=='compass_garden':
        axes=rotate(q[:,2:,:],np.broadcast_to([1.,0,0],q[:,2:,:3].shape))
        yaw=np.unwrap(np.arctan2(axes[:,:,1],axes[:,:,0]),axis=0)
        report['needle_yaw_range_degrees']=[float(v) for v in np.rad2deg(np.ptp(yaw,axis=0))]
        if full_demo:assert np.all(np.ptp(yaw,axis=0)>.1)
    elif cfg['scene']=='magnetic_crane':
        beads=x[:,3:];peak=int(np.argmax(np.sum(beads[:,:,2]>.25,axis=1)))
        report.update(peak_airborne_beads=int(np.sum(beads[peak,:,2]>.25)),peak_frame=peak,
            delivered_beads=int(np.sum(beads[-1,:,0]>.05)),max_bead_height=float(beads[:,:,2].max()),
            final_bead_bounds=[beads[-1].min(0).tolist(),beads[-1].max(0).tolist()])
        if full_demo:assert report['peak_airborne_beads']>=3 and report['delivered_beads']>=3
    else:
        triads=x[:,2:];report.update(peak_upper_hemisphere=int(np.max(np.sum(triads[:,:,2]>.47,axis=1))),
            peak_height=float(triads[:,:,2].max()),final_max_height=float(triads[-1,:,2].max()))
        if full_demo:assert report['peak_upper_hemisphere']>=3 and report['final_max_height']<.3
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('simulations',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--require-full-demo',action='store_true',help='Require the complete preset duration and scene interactions')
    a=p.parse_args();reports=[audit(f,a.require_full_demo) for f in a.simulations]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(passed=True,runs=reports),indent=2)+'\n')
    for r in reports:print(json.dumps({k:v for k,v in r.items() if k not in ['state_hashes','needle_yaw_range_degrees']}))

if __name__=='__main__':main()
