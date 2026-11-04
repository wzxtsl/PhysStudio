"""Audit every recorded 3D state, conservation report and frame interval."""
import argparse,json
from pathlib import Path
import numpy as np

def audit(path):
    cfg=json.loads((path/'config.json').read_text());done=json.loads((path/'complete.json').read_text());history=json.loads((path/'diagnostics.json').read_text())
    files=sorted((path/'states').glob('*.npz'));assert len(files)==cfg['frames']==done['frames']==len(history)
    initial=np.load(files[0]);previous=None;max_motion=0.;axis_ratios=[];min_h=float('inf');max_h=0.;max_mass_error=0.;max_soap_error=0.
    for i,file in enumerate(files):
        assert file.stem==f'{i:05d}';s=np.load(file)
        assert all(np.isfinite(s[k]).all() for k in s.files)
        assert abs(float(s['time'])-i*cfg['frame_dt'])<1e-9
        assert np.min(s['lh'])>0 and np.min(s['thickness'])>0 and np.min(s['gamma'])>0
        assert np.max(s['gamma'])<cfg['sigma0']/cfg['rt']
        max_mass_error=max(max_mass_error,abs(s['vol'].sum()/initial['vol'].sum()-1))
        max_soap_error=max(max_soap_error,abs(s['soap'].sum()/initial['soap'].sum()-1))
        min_h=min(min_h,float(s['lh'].min()));max_h=max(max_h,float(s['lh'].max()))
        if previous is not None:max_motion=max(max_motion,float(np.linalg.norm(s['x']-previous,axis=1).max()))
        previous=s['x']
        for group in range(len(cfg['bubbles'])):
            x=s['x'][s['eg']==group];ev=np.linalg.eigvalsh(np.cov(x.T));axis_ratios.append(float(np.sqrt(ev[-1]/ev[0])))
    assert max_mass_error<1e-12 and max_soap_error<1e-12
    assert max_motion<cfg['bubbles'][0]['radius']*.1,'Frame-to-frame displacement is excessive'
    for d in history[1:]:
        for key in ['normal','tangential']:assert d[key+'_residual']<=max(2*d[key+'_tolerance'],1e-13)
    max_vol=max(float(np.max(np.abs(np.array(d['enclosed_volume'])/d['initial_volume']-1))) for d in history)
    assert max_vol<.01,'Gas-volume variation exceeds the showcase tolerance'
    return dict(passed=True,simulation=str(path.resolve()),frames=len(files),physical_seconds=(len(files)-1)*cfg['frame_dt'],
        video_seconds=len(files)/cfg['fps'],e_particles=len(initial['x']),l_particles=len(initial['p']),
        material_liquid_relative_error=max_mass_error,material_surfactant_relative_error=max_soap_error,
        scatter_liquid_relative_error=max(d['liquid_transfer_relative_error'] for d in history),
        scatter_surfactant_relative_error=max(d['soap_transfer_relative_error'] for d in history),
        gas_volume_max_relative_change=max_vol,thickness_nm_range=[min_h*1e9,max_h*1e9],max_frame_displacement_m=max_motion,
        covariance_axis_ratio_range=[min(axis_ratios),max(axis_ratios)],
        max_projection_m=max(d['max_projection'] for d in history),max_penetration_m=max(0.,max(-d['min_signed_gap'] for d in history)),
        contact_points_range=[min(d['contact_points'] for d in history),max(d['contact_points'] for d in history)])

def main():
    p=argparse.ArgumentParser();p.add_argument('--simulation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.simulation);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
