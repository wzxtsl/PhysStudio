"""Measure rendered surface validity and its agreement with recorded material."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def main():
    p=argparse.ArgumentParser();p.add_argument('--simulation',type=Path,required=True);p.add_argument('--mesh',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=json.loads((a.simulation/'config.json').read_text());records=[]
    for file in sorted(a.mesh.glob('[0-9]*.npz')):
        data=np.load(file);state=np.load(a.simulation/'states'/file.name)
        x=data['vertices'].astype(float);f=data['faces'];g=data['groups'];h=data['thickness_nm'].astype(float)*1e-9
        tri=x[f];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=.5*np.linalg.norm(cross,axis=1)
        assert np.isfinite(x).all() and np.isfinite(h).all() and min(h)>0 and min(area)>1e-18
        for region in range(len(cfg['bubbles'])):
            mask=g[f[:,0]]==region;assert (g[f[mask]]==region).all()
            selected=state['eg']==region
            dist,nearest=cKDTree(state['x'][selected]).query(tri[mask].mean(1))
            flipped=np.sum(cross[mask]*state['n'][selected][nearest],axis=1)<0
            physical=state['vol'][state['lg']==region].sum()
            integrated=np.sum(area[mask]*h[f[mask]].mean(1))
            gas=np.sum(np.einsum('ij,ij->i',tri[mask,0],cross[mask]))/6
            records.append(dict(frame=int(file.stem),region=region,flipped_triangles=int(flipped.sum()),triangles=int(mask.sum()),min_area_m2=float(area[mask].min()),liquid_volume_ratio=float(integrated/physical),mesh_gas_volume_m3=float(gas),surface_area_m2=float(area[mask].sum()),thickness_nm_range=[float(h[g==region].min()*1e9),float(h[g==region].max()*1e9)]))
    assert records
    report=dict(frames=len(records)//len(cfg['bubbles']),max_flipped_triangles=max(r['flipped_triangles'] for r in records),liquid_volume_ratio_range=[min(r['liquid_volume_ratio'] for r in records),max(r['liquid_volume_ratio'] for r in records)],records=records)
    report['passed']=report['max_flipped_triangles']==0 and report['liquid_volume_ratio_range'][0]>.8 and report['liquid_volume_ratio_range'][1]<1.2
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2));assert report['passed']
if __name__=='__main__':main()
