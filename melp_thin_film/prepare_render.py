"""Reconstruct a smooth visual surface from recorded MELP geometry/material points."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree,ConvexHull
import trimesh


def project(points,x,n,h):
    dist,ids=cKDTree(x).query(points,k=min(32,len(x)),workers=4)
    w=np.maximum(0,1-dist/h)**4*(1+4*dist/h)
    nn=np.einsum('ij,ijk->ik',w,n[ids]);nn/=np.linalg.norm(nn,axis=1)[:,None]
    axis=np.zeros_like(nn);axis[:,2]=1;mask=np.abs(nn[:,2])>.85;axis[mask]=[0,1,0]
    a=np.cross(axis,nn);a/=np.linalg.norm(a,axis=1)[:,None];b=np.cross(nn,a)
    d=(x[ids]-points[:,None,:])/h
    u=np.einsum('ijk,ik->ij',d,a);v=np.einsum('ijk,ik->ij',d,b);z=np.einsum('ijk,ik->ij',d,nn)
    f=np.stack([np.ones_like(u),u,v,u*u,u*v,v*v],axis=-1)
    A=np.einsum('nk,nki,nkj->nij',w,f,f)+np.eye(6)[None]*1e-9
    rhs=np.einsum('nk,nki,nk->ni',w,f,z)
    c=np.linalg.solve(A,rhs)
    return points+h*c[:,0,None]*nn


def main():
    p=argparse.ArgumentParser();p.add_argument('--simulation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--indices');p.add_argument('--subdivision',type=int,default=2);p.add_argument('--resume',action='store_true');p.add_argument('--material-density',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);cfg=json.loads((a.simulation/'config.json').read_text());files=sorted((a.simulation/'states').glob('*.npz'))
    indices=[int(i) for i in a.indices.split(',')] if a.indices else range(len(files))
    if (a.output/'config.json').exists():
        assert json.loads((a.output/'config.json').read_text())==cfg,'Existing reconstruction uses another configuration'
    if a.resume:
        for metadata in a.output.glob('reconstruction*.json'):
            previous=json.loads(metadata.read_text())
            assert previous['subdivision']==a.subdivision and previous['simulation']==str(a.simulation.resolve())
            assert previous.get('material_density',False)==a.material_density
    initial=np.load(files[0]);topology={}
    for g in range(len(cfg['bubbles'])):
        x=initial['x'][initial['eg']==g];hull=ConvexHull(x);faces=hull.simplices.copy()
        cross=np.cross(x[faces[:,1]]-x[faces[:,0]],x[faces[:,2]]-x[faces[:,0]])
        flip=np.sum(cross*hull.equations[:,:3],axis=1)<0
        faces[flip]=faces[flip][:,[0,2,1]];topology[g]=faces
    for i in indices:
        if a.resume and (a.output/f'{i:05d}.npz').exists():continue
        data=np.load(files[i]);vertices=[];faces_all=[];thickness=[];groups=[];offset=0
        for g in range(len(cfg['bubbles'])):
            e=data['eg']==g;l=data['lg']==g;x=data['x'][e];n=data['n'][e]
            faces=topology[g].copy();vertices_g=x.copy()
            for _ in range(a.subdivision):vertices_g,faces=trimesh.remesh.subdivide(vertices_g,faces)
            support=cfg['support']*np.sqrt(4*np.pi*cfg['bubbles'][g]['radius']**2/cfg['e_count'])
            vertices_g=project(vertices_g,x,n,support)
            material_tree=cKDTree(data['p'][l]);dist,ids=material_tree.query(vertices_g,k=12,workers=4)
            w=np.exp(-3*(dist/np.maximum(dist[:,-1,None],1e-12))**2);w/=w.sum(axis=1)[:,None]
            material_h=data['lh'][l]
            if a.material_density:
                # Reconstruct thickness from conserved liquid volume and the
                # actual fine-particle number density, including contact regions.
                # This uses no display-only animation or thickness clamp.
                h_support=3*np.sqrt(4*np.pi*cfg['bubbles'][g]['radius']**2/np.sum(l))
                pairs=material_tree.sparse_distance_matrix(material_tree,h_support,output_type='coo_matrix')
                q=pairs.data/h_support;kernel=(1-q)**4*(1+4*q)*7/(np.pi*h_support*h_support)
                density=np.bincount(pairs.row,weights=kernel,minlength=np.sum(l))
                material_h=data['vol'][l]*density
            h=np.sum(w*material_h[ids],axis=1)
            # Each closed lamella is a display mesh; this mesh never enters the solver.
            vertices.append(vertices_g);thickness.append(h);faces_all.append(faces+offset);groups.extend([g]*len(vertices_g));offset+=len(vertices_g)
        np.savez_compressed(a.output/f'{i:05d}.npz',vertices=np.concatenate(vertices).astype(np.float32),faces=np.concatenate(faces_all).astype(np.int32),thickness_nm=(1e9*np.concatenate(thickness)).astype(np.float32),groups=np.array(groups,np.int32),time=data['time'])
        if i%30==0:print('Prepared',i,flush=True)
    (a.output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    metadata_name='reconstruction.json' if not a.indices else f'reconstruction_{indices[0]:05d}_{indices[-1]:05d}.json'
    (a.output/metadata_name).write_text(json.dumps(dict(indices=list(indices),subdivision=a.subdivision,material_density=a.material_density,method='Quadratic MLS geometry and material-thickness interpolation',simulation=str(a.simulation.resolve())),indent=2)+'\n')

if __name__=='__main__':main()
