"""Continuous ice/water interfaces from particle enthalpy and volume."""
import argparse,json,time
from pathlib import Path
import numpy as np
import warp as wp
from implicit_mixture.surface import smooth_surface,grid_to_world_vertices
from ice_melting.thermal import fraction
from .boundaries import star_distance,distance_and_reflection
from .phase_texture import pack_phase_volume


@wp.kernel(enable_backward=False)
def deposit(x:wp.array(dtype=wp.vec3),phase:wp.array(dtype=float),field:wp.array3d(dtype=wp.vec2),
            origin:wp.vec3,ds:float,dx:float):
    i=wp.tid();p=x[i]
    base=wp.vec3i(int(wp.floor((p[0]-origin[0])/ds)),int(wp.floor((p[1]-origin[1])/ds)),int(wp.floor((p[2]-origin[2])/ds)))
    reach=int(wp.ceil(2.4*dx/ds));f=phase[i]
    volume=wp.vec2(1.-f,.917*f)
    for a in range(-reach,reach+1):
        for b in range(-reach,reach+1):
            for c in range(-reach,reach+1):
                index=base+wp.vec3i(a,b,c)
                if index[0]>=0 and index[1]>=0 and index[2]>=0 and index[0]<field.shape[0] and index[1]<field.shape[1] and index[2]<field.shape[2]:
                    r=(origin+ds*wp.vec3(float(index[0]),float(index[1]),float(index[2]))-p)/dx
                    r2=wp.dot(r,r)
                    if r2<2.4*2.4:
                        w=wp.exp(-r2/(2.*.8*.8))/(wp.pow(2.*3.141592653589793,1.5)*.8*.8*.8)
                        wp.atomic_add(field,index[0],index[1],index[2],volume*w)


@wp.kernel(enable_backward=False)
def levelset(field:wp.array3d(dtype=wp.vec2),iso:wp.array3d(dtype=float),which:int,
             origin:wp.vec3,ds:float,dx:float,mode:int,bowl:float,rod:wp.vec3,rod_r:float):
    i,j,k=wp.tid();v=field[i,j,k];total=v[0]+v[1];ice=v[0]/wp.max(total,1.e-8)
    phase=ice-.5
    if which==1:phase=.5-ice
    value=wp.min(total-.46,phase)
    if which==2:value=total-.46
    p=origin+ds*wp.vec3(float(i),float(j),float(k))
    value=wp.min(value,(p[2]+.002*dx)/dx)
    if mode==1:
        r=wp.length(wp.vec3(p[0],p[1],0.));dr=wp.length(wp.vec3(p[0]-rod[0],p[1]-rod[1],0.))
        value=wp.min(value,(bowl+.002*dx-r)/dx)
        value=wp.min(value,(dr-rod_r+.002*dx)/dx)
    elif mode==3:
        value=wp.min(value,(-star_distance(wp.vec2(p[0],p[1]),bowl)+.002*dx)/dx)
    elif mode==2:
        # Intersection with the complement of the two cooled stair blocks.
        value=wp.min(value,wp.max((p[0]+.025+.002*dx)/dx,(p[2]-.026+.002*dx)/dx))
        value=wp.min(value,wp.max((p[0]-.025+.002*dx)/dx,(p[2]-.014+.002*dx)/dx))
    iso[i,j,k]=value


def support_samples(x,f,cfg):
    dx=cfg['dx'];support=2.4*dx
    parts=[x];phases=[f]
    if cfg['boundary_mode']==3:
        d,reflected=distance_and_reflection(x,cfg['star_radius'])
        mask=d>-support;parts.append(reflected[mask]);phases.append(f[mask])
    if cfg['boundary_mode']==1:
        r=np.linalg.norm(x[:,:2],axis=1);radius=cfg['bowl_radius']
        mask=r>radius-support;wall=x[mask].copy()
        wall[:,:2]*=((2*radius-r[mask])/r[mask])[:,None];parts.append(wall);phases.append(f[mask])
        center=np.array(cfg['rod_center']);d=x-center;r=np.linalg.norm(d[:,:2],axis=1);radius=cfg['rod_radius']
        mask=(r<radius+min(support,radius*.8))&(r>0)
        wall=x[mask].copy();wall[:,:2]=center[:2]+d[mask,:2]*((2*radius-r[mask])/r[mask])[:,None]
        parts.append(wall);phases.append(f[mask])
    if cfg['boundary_mode']==2:
        for lo,hi,top in [(-.090,-.025,.026),(-.025,.025,.014)]:
            mask=(x[:,0]>=lo)&(x[:,0]<=hi)&(x[:,2]<top+support)&(x[:,2]>=top)
            q=x[mask].copy();q[:,2]=2*top-q[:,2];parts.append(q);phases.append(f[mask])
            mask=(x[:,0]>hi)&(x[:,0]<hi+support)&(x[:,2]<top)
            q=x[mask].copy();q[:,0]=2*hi-q[:,0];parts.append(q);phases.append(f[mask])
    x=np.concatenate(parts);f=np.concatenate(phases)
    mask=x[:,2]<support;q=x[mask].copy();q[:,2]*=-1
    return np.concatenate([x,q]),np.r_[f,f[mask]]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,nargs='*')
    p.add_argument('--device',default='cuda:0');p.add_argument('--scale',type=float,default=.42)
    args=p.parse_args();cfg=json.loads((args.input/'config.json').read_text());indices=args.frames or list(range(cfg['frames']))
    args.output.mkdir(parents=True,exist_ok=False);dx=cfg['dx'];ds=dx*args.scale
    origin=np.array(cfg['bounds'][0])-3*dx;upper=np.array(cfg['bounds'][1])+3*dx
    upper[2]=max(float(np.load(args.input/'states'/f'{i:05d}.npz')['positions'][:,2].max()) for i in indices)+3*dx
    shape=tuple((np.ceil((upper-origin)/ds).astype(int)+1).tolist());wp.init()
    field=wp.zeros(shape,dtype=wp.vec2,device=args.device);iso=wp.zeros(shape,dtype=float,device=args.device)
    mc=wp.MarchingCubes(*shape,max_verts=1500000,max_tris=3000000,device=args.device)
    records=[];start=time.monotonic()
    for index in indices:
        state=np.load(args.input/'states'/f'{index:05d}.npz');x,f=support_samples(state['positions'],fraction(state['enthalpy']),cfg)
        field.zero_();xgpu=wp.array(x,dtype=wp.vec3,device=args.device);fgpu=wp.array(f,dtype=float,device=args.device)
        wp.launch(deposit,len(x),inputs=[xgpu,fgpu,field,wp.vec3(*origin),ds,dx],device=args.device)
        data={};stats={}
        continuous=cfg.get('phase_rendering')=='continuous'
        surfaces=[(2,'fluid')] if continuous else [(0,'ice'),(1,'water')]
        for k,name in surfaces:
            wp.launch(levelset,shape,inputs=[field,iso,k,wp.vec3(*origin),ds,dx,cfg['boundary_mode'],
                cfg.get('star_radius',cfg.get('bowl_radius',0.)),wp.vec3(*cfg.get('rod_center',[0.,0.,0.])),cfg.get('rod_radius',0.)],device=args.device)
            mc.surface(iso,0.)
            v=grid_to_world_vertices(mc.verts.numpy(),ds,origin) if mc.verts.size else np.empty((0,3),np.float32)
            faces=mc.indices.numpy().reshape(-1,3)[:,[0,2,1]] if mc.indices.size else np.empty((0,3),np.int32)
            v,q=smooth_surface(v,faces,4)
            data[name+'_vertices']=v;data[name+'_faces']=faces;stats[name]=dict(triangles=len(faces),smoothing=q)
        if continuous:
            values=field.numpy();total=values.sum(axis=-1)
            phase=np.clip(values[...,0]/np.maximum(total,1.e-8),0.,1.)
            atlas,columns=pack_phase_volume(phase)
            data.update(phase_atlas=atlas,phase_columns=columns,phase_shape=shape,
                        phase_origin=origin,phase_spacing=ds)
        np.savez_compressed(args.output/f'{index:05d}.npz',**data)
        records.append(dict(frame=index,stats=stats))
        if index%30==0 or index==indices[-1]:print(json.dumps(dict(frame=index,elapsed=time.monotonic()-start,stats=stats)),flush=True)
    (args.output/'manifest.json').write_text(json.dumps(dict(frames=indices,spacing=ds,records=records),indent=2))


if __name__=='__main__':main()
