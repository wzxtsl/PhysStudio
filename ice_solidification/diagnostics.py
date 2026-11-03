"""Audit mass, heat, stationary ice, and collider agreement across a trajectory."""
import argparse,json
from pathlib import Path
import numpy as np
from .geometry import distance_and_reflection


def audit(folder):
    cfg=json.loads((folder/'config.json').read_text());rows=json.loads((folder/'audit.json').read_text())
    files=sorted((folder/'states').glob('*.npz'))
    assert len(files)==cfg['frames']==len(rows)
    anchors={};drift=0.;penetration=0.;phase_increases=[];last_ice=0.
    radius=cfg['dx']*.46
    for frame,file in enumerate(files):
        state=np.load(file);x=state['positions'];g=state['groups']
        assert len(x)==cfg['particles']
        assert all(np.isfinite(state[k]).all() for k in state.files)
        for index in np.flatnonzero(g>=0):
            if index in anchors:drift=max(drift,float(np.linalg.norm(x[index]-anchors[index])))
            else:anchors[index]=x[index].copy()
        penetration=max(penetration,float(np.maximum(radius-x[:,2],0).max()))
        mode=cfg['boundary_mode']
        if mode==1:
            radial=np.linalg.norm(x[:,:2],axis=1)
            rod=np.linalg.norm(x[:,:2]-np.array(cfg['rod_center'])[:2],axis=1)
            penetration=max(penetration,float(np.maximum(radial-cfg['bowl_radius']+radius,0).max()),
                            float(np.maximum(cfg['rod_radius']+radius-rod,0).max()))
        elif mode==2:
            for edge,top in [(-.025,.026),(.025,.014)]:
                depth=np.minimum(edge+radius-x[:,0],top+radius-x[:,2])
                penetration=max(penetration,float(np.maximum(depth,0).max()))
        elif mode==3:
            d,_=distance_and_reflection(x,cfg['star_radius'])
            penetration=max(penetration,float(np.maximum(d+radius,0).max()))
        phase_increases.append(rows[frame]['ice_fraction']-last_ice);last_ice=rows[frame]['ice_fraction']
    heat=max(abs(r['relative_energy_balance']) for r in rows)
    volume=rows[-1]['physical_volume']/rows[0]['physical_volume']-1.
    expected=rows[-1]['ice_fraction']*(1000/917-1.)
    report=dict(scene=cfg['scene'],frames=len(files),particles=cfg['particles'],
        mass_kg=cfg['mass']*cfg['particles'],max_relative_heat_error=heat,
        final_ice_fraction=last_ice,volume_increase=volume,constitutive_volume_increase=expected,
        maximum_anchor_drift_m=drift,maximum_collider_penetration_m=penetration,
        maximum_speed=max(r['max_speed'] for r in rows),
        thermal_seconds=cfg['seconds']*cfg['thermal_scale'],mechanical_seconds=cfg['seconds'])
    assert heat<5.e-4 and drift<1.e-7 and penetration<cfg['dx']*.02,report
    assert abs(volume-expected)<2.e-6,report
    report['passed']=True
    return report,rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result,rows=audit(args.input);args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'audit.json').write_text(json.dumps(result,indent=2))
    import csv
    with (args.output/'curves.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    print(json.dumps(result,indent=2))
    try:import matplotlib
    except ImportError:return
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.3),layout='constrained')
    t=[r['time'] for r in rows]
    axes[0].plot(t,[r['ice_fraction']*100 for r in rows]);axes[0].set_ylabel('Solid fraction (%)')
    axes[1].plot(t,[r['relative_energy_balance'] for r in rows]);axes[1].set_ylabel('Heat residual / total latent heat')
    axes[2].plot(t,[r['rms_liquid_speed'] for r in rows]);axes[2].set_ylabel('Liquid RMS speed (m/s)')
    for ax in axes:ax.set_xlabel('Mechanical time (s)');ax.grid(alpha=.2)
    fig.savefig(args.output/'curves.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
