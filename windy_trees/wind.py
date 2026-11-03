"""Weakly compressible SPH air and equal/opposite branch/leaf drag impulses."""
import numpy as np
import warp as wp
from .tree import exp_rotation

@wp.func
def W(distance:float,h:float):
    q=distance/h
    value=float(0.0)
    if q<.5:
        value=1.0+6.0*(q*q*q-q*q)
    elif q<1.0:
        value=2.0*wp.pow(1.0-q,3.0)
    return value*8.0/(wp.pi*h*h*h)

@wp.func
def G(r:wp.vec3,h:float):
    d=wp.length(r);q=d/h;a=float(0.0)
    if d>1.e-8 and q<1.0:
        if q<.5:a=6.0*(3.0*q*q-2.0*q)
        else:a=-6.0*(1.0-q)*(1.0-q)
        return r*(a*8.0/(wp.pi*d*h*h*h*h))
    return wp.vec3(0.0)

@wp.func
def background(p:wp.vec3,t:float,speed:float,turbulence:float,direction:float,turn:float):
    angle=direction+turn*wp.sin(.62*t-.4)
    gust=.78+.26*wp.sin(1.65*t-.7)+.16*wp.sin(3.35*t+.3)
    ramp=1.0-wp.exp(-t/0.45)
    profile=wp.pow(wp.clamp((p[2]+.12)/3.0,.04,2.2),.16)
    main=wp.vec3(wp.cos(angle),wp.sin(angle),0.0)*(speed*gust*profile)
    # Each perturbation component is independent of its own coordinate.
    # This divergence-free inlet signal supplies eddies transported by SPH.
    u=wp.sin(1.4*p[2]-2.3*t)+.4*wp.cos(2.1*p[1]+1.7*t)
    v=.8*wp.sin(1.05*p[0]-1.8*t)+.45*wp.cos(2.0*p[2]+2.8*t)
    w=.55*wp.sin(1.7*p[0]-2.6*t)+.35*wp.sin(1.5*p[1]+2.1*t)
    return ramp*(main+turbulence*wp.vec3(u,v,w))

@wp.kernel(enable_backward=False)
def density(x:wp.array(dtype=wp.vec3),grid:wp.uint64,mass:float,h:float,
            rho0:float,sound:float,rho:wp.array(dtype=float),pressure:wp.array(dtype=float)):
    i=wp.tid();rr=float(0.0)
    for j in wp.hash_grid_query(grid,x[i],h):rr+=mass*W(wp.length(x[i]-x[j]),h)
    rho[i]=wp.max(rr,.2*rho0)
    pressure[i]=sound*sound*wp.max(rr-rho0,0.0)

@wp.kernel(enable_backward=False)
def fluid_forces(x:wp.array(dtype=wp.vec3),v:wp.array(dtype=wp.vec3),grid:wp.uint64,
                 rho:wp.array(dtype=float),pressure:wp.array(dtype=float),mass:float,h:float,
                 nu:float,sound:float,t:float,speed:float,turbulence:float,direction:float,turn:float,
                 lo:wp.vec3,hi:wp.vec3,acc:wp.array(dtype=wp.vec3)):
    i=wp.tid();point=x[i];vel=v[i];a=wp.vec3(0.0)
    for j in wp.hash_grid_query(grid,point,h):
        if j!=i:
            r=point-x[j];g=G(r,h);dv=vel-v[j]
            pressure_pair=pressure[i]/(rho[i]*rho[i])+pressure[j]/(rho[j]*rho[j])
            approach=wp.dot(dv,r)
            artificial=float(0.0)
            if approach<0.0:
                artificial=-.08*sound*h*approach/((wp.dot(r,r)+.01*h*h)*.5*(rho[i]+rho[j]))
            a-=mass*(pressure_pair+artificial)*g
            lap=-2.0*wp.dot(r,g)/(wp.dot(r,r)+.01*h*h)
            a+=nu*2.0*mass/(rho[i]+rho[j])*lap*(v[j]-vel)
    target=background(point,t,speed,turbulence,direction,turn)
    rate=float(.25)
    if point[0]<lo[0]+1.3 or point[1]<lo[1]+.65 or point[1]>hi[1]-.65:
        rate=5.0
    a+=rate*(target-vel)
    acc[i]=a

@wp.func
def sample_air(point:wp.vec3,x:wp.array(dtype=wp.vec3),v:wp.array(dtype=wp.vec3),grid:wp.uint64,h:float):
    sumv=wp.vec3(0.0);weight=float(0.0)
    for j in wp.hash_grid_query(grid,point,h):
        w=W(wp.length(point-x[j]),h);sumv+=w*v[j];weight+=w
    return wp.vec4(sumv[0],sumv[1],sumv[2],weight)

@wp.func
def reaction(point:wp.vec3,force:wp.vec3,weight:float,x:wp.array(dtype=wp.vec3),
             grid:wp.uint64,h:float,air_force:wp.array(dtype=wp.vec3)):
    if weight>1.e-9:
        for j in wp.hash_grid_query(grid,point,h):
            w=W(wp.length(point-x[j]),h)/weight
            wp.atomic_add(air_force,j,-w*force)

@wp.kernel(enable_backward=False)
def branch_drag(p:wp.array(dtype=wp.vec3),v:wp.array(dtype=wp.vec3),parent:wp.array(dtype=int),
                radius:wp.array(dtype=float),x:wp.array(dtype=wp.vec3),air_v:wp.array(dtype=wp.vec3),
                grid:wp.uint64,h:float,rho:float,cd:float,feedback:int,
                force:wp.array(dtype=wp.vec3),moment:wp.array(dtype=wp.vec3),air_force:wp.array(dtype=wp.vec3)):
    i=wp.tid()
    if i>0:
        par=parent[i];edge=p[i]-p[par];length=wp.length(edge);axis=edge/wp.max(length,1.e-8)
        point=(p[i]+p[par])*.5;sample=sample_air(point,x,air_v,grid,h)
        if sample[3]>1.e-9:
            relative=wp.vec3(sample[0],sample[1],sample[2])/sample[3]-.5*(v[i]+v[par])
            transverse=relative-wp.dot(relative,axis)*axis
            f=.5*rho*cd*(2.0*radius[i]*length)*wp.length(transverse)*transverse
            wp.atomic_add(force,i,f);wp.atomic_add(moment,i,wp.cross(point,f))
            if feedback==1:reaction(point,f,sample[3],x,grid,h,air_force)

@wp.kernel(enable_backward=False)
def leaf_drag(owner:wp.array(dtype=int),direction:wp.array(dtype=wp.vec3),across:wp.array(dtype=wp.vec3),
              normal:wp.array(dtype=wp.vec3),size:wp.array(dtype=float),area:wp.array(dtype=float),
              angle:wp.array(dtype=float),rate:wp.array(dtype=float),p:wp.array(dtype=wp.vec3),
              v:wp.array(dtype=wp.vec3),q:wp.array(dtype=wp.quat),x:wp.array(dtype=wp.vec3),
              air_v:wp.array(dtype=wp.vec3),grid:wp.uint64,h:float,rho:float,cd:float,feedback:int,
              force:wp.array(dtype=wp.vec3),moment:wp.array(dtype=wp.vec3),
              air_force:wp.array(dtype=wp.vec3),torque:wp.array(dtype=float),air_damping:wp.array(dtype=float),local_wind:wp.array(dtype=wp.vec3)):
    i=wp.tid();body=owner[i];rot=q[body]*wp.quat_from_axis_angle(across[i],angle[i])
    hinge=wp.quat_rotate(q[body],across[i])
    lever=wp.quat_rotate(rot,direction[i])*(.025+.45*size[i])
    point=p[body]+lever;nn=wp.quat_rotate(rot,normal[i]);sample=sample_air(point,x,air_v,grid,h)
    if sample[3]>1.e-9:
        wind=wp.vec3(sample[0],sample[1],sample[2])/sample[3]
        relative=wind-v[body]-wp.cross(hinge*rate[i],lever)
        vn=wp.dot(relative,nn);tangent=relative-vn*nn
        f=.5*rho*cd*area[i]*(vn*wp.abs(vn)*nn+.12*wp.length(tangent)*tangent)
        wp.atomic_add(force,body,f);wp.atomic_add(moment,body,wp.cross(point,f))
        torque[i]=wp.dot(wp.cross(lever,f),hinge);local_wind[i]=wind
        sensitivity=wp.cross(hinge,lever);sn=wp.dot(sensitivity,nn)
        st=sensitivity-sn*nn;vt=wp.length(tangent)
        air_damping[i]=.5*rho*cd*area[i]*(2.0*wp.abs(vn)*sn*sn+.12*(vt*wp.dot(st,st)+wp.pow(wp.dot(tangent,st),2.0)/wp.max(vt,1.e-6)))
        if feedback==1:reaction(point,f,sample[3],x,grid,h,air_force)

@wp.kernel(enable_backward=False)
def advect(x:wp.array(dtype=wp.vec3),v:wp.array(dtype=wp.vec3),acc:wp.array(dtype=wp.vec3),
           force:wp.array(dtype=wp.vec3),mass:float,dt:float,lo:wp.vec3,hi:wp.vec3,
           t:float,speed:float,turbulence:float,direction:float,turn:float):
    i=wp.tid();vel=v[i]+dt*(acc[i]+force[i]/mass);point=x[i]+dt*vel
    recycled=bool(False)
    for axis in range(3):
        if point[axis]<lo[axis] or point[axis]>hi[axis]:
            length=hi[axis]-lo[axis]
            point[axis]=lo[axis]+(point[axis]-lo[axis])-wp.floor((point[axis]-lo[axis])/length)*length
            recycled=True
    if recycled:vel=background(point,t,speed,turbulence,direction,turn)
    x[i]=point;v[i]=vel

class Wind:
    def __init__(self,cfg,device='cuda:0'):
        self.cfg=cfg;self.device=device
        lo,hi=np.asarray(cfg['bounds']);spacing=cfg['spacing'];self.h=cfg['support']
        coords=[np.arange(lo[k]+spacing*.5,hi[k],spacing) for k in range(3)]
        xyz=np.stack(np.meshgrid(*coords,indexing='ij'),-1).reshape(-1,3).astype(np.float32)
        offset=np.arange(-int(np.ceil(self.h/spacing)),int(np.ceil(self.h/spacing))+1)*spacing
        r=np.linalg.norm(np.stack(np.meshgrid(offset,offset,offset),-1),axis=-1)/self.h
        w=np.where(r<.5,1+6*(r**3-r*r),np.where(r<1,2*(1-r)**3,0))*8/(np.pi*self.h**3)
        self.mass=cfg['air_density']/float(w.sum());self.n=len(xyz)
        self.lo=wp.vec3(*map(float,lo));self.hi=wp.vec3(*map(float,hi))
        self.x=wp.array(xyz,dtype=wp.vec3,device=device);self.v=wp.zeros_like(self.x)
        self.acc=wp.zeros_like(self.x);self.force=wp.zeros_like(self.x)
        self.rho=wp.zeros(self.n,dtype=float,device=device);self.pressure=wp.zeros_like(self.rho)
        self.grid=wp.HashGrid(64,48,40,device=device);self.grid.reserve(self.n)
        self.local_wind=None

    def begin(self,t):
        c=self.cfg;self.force.zero_();self.grid.build(self.x,self.h)
        wp.launch(density,self.n,[self.x,self.grid.id,self.mass,self.h,c['air_density'],c['sound_speed'],self.rho,self.pressure],device=self.device)
        wp.launch(fluid_forces,self.n,[self.x,self.v,self.grid.id,self.rho,self.pressure,self.mass,self.h,
            c['viscosity'],c['sound_speed'],t,c['wind_speed'],c['turbulence'],c['direction'],c['turn'],self.lo,self.hi,self.acc],device=self.device)

    def couple(self,tree):
        c=self.cfg
        if self.local_wind is None:self.local_wind=wp.zeros(tree.nl,dtype=wp.vec3,device=self.device)
        wp.launch(branch_drag,tree.n,[tree.p,tree.v,tree.parent,tree.radius,self.x,self.v,self.grid.id,self.h,
                  c['air_density'],c['drag_wood'],int(c['coupling']),tree.force,tree.moment,self.force],device=self.device)
        wp.launch(leaf_drag,tree.nl,[tree.leaf_owner,tree.leaf_direction,tree.leaf_across,tree.leaf_normal,
                  tree.leaf_size,tree.leaf_area,tree.leaf_theta,tree.leaf_omega,tree.p,tree.v,tree.q,
                  self.x,self.v,self.grid.id,self.h,c['air_density'],c['drag_leaf'],int(c['coupling']),
                  tree.force,tree.moment,self.force,tree.leaf_torque,tree.leaf_air_damping,self.local_wind],device=self.device)

    def advance(self,dt,t):
        c=self.cfg
        wp.launch(advect,self.n,[self.x,self.v,self.acc,self.force,self.mass,dt,self.lo,self.hi,t,
                  c['wind_speed'],c['turbulence'],c['direction'],c['turn']],device=self.device)
