"""Audit dense wind filaments against the recorded SPH data and physical presets."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .flow import Pathlines,WindStreaks


def inspect(folder,baseline=None):
    cfg=json.loads((folder/'config.json').read_text())
    provenance=json.loads((folder/'provenance.json').read_text())
    changed={}
    if baseline is not None:
        old=json.loads((baseline/'config.json').read_text())
        changed={k:[old.get(k),cfg.get(k)] for k in sorted(set(old)|set(cfg)) if old.get(k)!=cfg.get(k)}
        assert set(changed)<=set(['trace_count']),changed
        original=json.loads((baseline/'provenance.json').read_text())
        assert provenance['state_sha256']==original['state_sha256']
    ids=np.load(folder/'trace_particle_ids.npy');assert len(np.unique(ids))==len(ids)
    paths=WindStreaks(folder/'states',cfg['fps']);curves=0;old_segments=0;max_error=0.;max_step=0.
    total=len(list((folder/'states').glob('*.npz')))
    assert total>=2
    sampled_frames=sorted(set([1]+np.linspace(1,total-1,min(11,total-1),dtype=int).tolist()))
    for frame in sampled_frames:
        for curve in paths.curves(frame):
            p,_=paths.state(curve['last_frame'])
            max_error=max(max_error,float(np.linalg.norm(curve['points'][-1]-p[curve['particle']])))
            first,_=paths.state(curve['first_frame'])
            max_error=max(max_error,float(np.linalg.norm(curve['points'][0]-first[curve['particle']])))
            max_step=max(max_step,float(np.linalg.norm(np.diff(curve['points'],axis=0),axis=1).max()))
            assert np.isfinite(curve['points']).all()
            assert np.all((curve['fade']>=0)&(curve['fade']<=1))
            curves+=1;old_segments+=int(curve['last_frame']<frame)
    assert max_error<1.e-6 and max_step<1.,(max_error,max_step)
    sparse=Pathlines(folder/'states',cfg['fps']);assert len(sparse.state(0)[0])<=450
    return dict(scene=cfg['scene'],changed_configuration=changed,recorded_particles=len(ids),
                checked_curves=curves,retained_pre_recycle_segments=old_segments,
                max_endpoint_error_m=max_error,max_interpolated_step_m=max_step,
                old_color_style_particles=len(sparse.state(0)[0]),sampled_frames=sampled_frames,
                source_tree_sha256=provenance['state_sha256'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--simulations',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    results={folder.name:inspect(folder) for folder in args.simulations}
    results['implementation_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),Path(__file__).with_name('flow.py')]}
    (args.output/'flow_validation.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
