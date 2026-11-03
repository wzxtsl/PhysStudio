"""Length-preserving articulated branches with implicit rotational elasticity."""
import numpy as np
import warp as wp

@wp.func
def exp_rotation(v:wp.vec3):
    angle=wp.length(v)
    if angle<1.e-10:
        return wp.quat_identity()
    return wp.quat_from_axis_angle(v/angle,angle)

@wp.kernel(enable_backward=False)
def global_rotations(theta:wp.array(dtype=wp.vec3),parent:wp.array(dtype=int),q:wp.array(dtype=wp.quat)):
    i=wp.tid();j=i;rotation=wp.quat_identity()
    while j>0:
        rotation=exp_rotation(theta[j])*rotation
        j=parent[j]
    q[i]=wp.normalize(rotation)

@wp.kernel(enable_backward=False)
def forward(parent:wp.array(dtype=int),edge:wp.array(dtype=wp.vec3),q:wp.array(dtype=wp.quat),
            p:wp.array(dtype=wp.vec3),v:wp.array(dtype=wp.vec3),base:wp.vec3,dt:float):
    i=wp.tid();j=i;point=base
    while j>0:
        point+=wp.quat_rotate(q[j],edge[j])
        j=parent[j]
    v[i]=(point-p[i])/dt
    p[i]=point

@wp.kernel(enable_backward=False)
def accumulate_load(parent:wp.array(dtype=int),force:wp.array(dtype=wp.vec3),moment:wp.array(dtype=wp.vec3),
                    total_force:wp.array(dtype=wp.vec3),total_moment:wp.array(dtype=wp.vec3)):
    i=wp.tid();j=i
    while j>=0:
        wp.atomic_add(total_force,j,force[i]);wp.atomic_add(total_moment,j,moment[i])
        j=parent[j]

@wp.kernel(enable_backward=False)
def cg_initialize(parent:wp.array(dtype=int),p:wp.array(dtype=wp.vec3),q:wp.array(dtype=wp.quat),
              total_force:wp.array(dtype=wp.vec3),total_moment:wp.array(dtype=wp.vec3),
              inertia:wp.array(dtype=float),stiffness:wp.array(dtype=float),damping:wp.array(dtype=float),
              theta:wp.array(dtype=wp.vec3),omega:wp.array(dtype=wp.vec3),dt:float,
              residual:wp.array(dtype=wp.vec3),z:wp.array(dtype=wp.vec3),direction:wp.array(dtype=wp.vec3)):
    i=wp.tid()
    if i>0:
        par=parent[i]
        world_torque=total_moment[i]-wp.cross(p[par],total_force[i])
        torque=wp.quat_rotate_inv(q[par],world_torque)
        r=dt*(torque-stiffness[i]*theta[i])-(dt*damping[i]+dt*dt*stiffness[i])*omega[i]
        residual[i]=r;z[i]=r/(inertia[i]+dt*damping[i]+dt*dt*stiffness[i]);direction[i]=z[i]

@wp.kernel(enable_backward=False)
def mass_action(parent:wp.array(dtype=int),p:wp.array(dtype=wp.vec3),q:wp.array(dtype=wp.quat),
                mass:wp.array(dtype=float),vector:wp.array(dtype=wp.vec3),
                force:wp.array(dtype=wp.vec3),moment:wp.array(dtype=wp.vec3)):
    i=wp.tid();j=i;velocity=wp.vec3(0.0)
    while j>0:
        par=parent[j]
        w=wp.quat_rotate(q[par],vector[j])
        velocity+=wp.cross(w,p[i]-p[par])
        j=par
    f=mass[i]*velocity;force[i]=f;moment[i]=wp.cross(p[i],f)

@wp.kernel(enable_backward=False)
def matrix_finish(parent:wp.array(dtype=int),p:wp.array(dtype=wp.vec3),q:wp.array(dtype=wp.quat),
                  total_force:wp.array(dtype=wp.vec3),total_moment:wp.array(dtype=wp.vec3),
                  stiffness:wp.array(dtype=float),damping:wp.array(dtype=float),dt:float,
                  vector:wp.array(dtype=wp.vec3),result:wp.array(dtype=wp.vec3)):
    i=wp.tid()
    if i>0:
        par=parent[i];torque=wp.quat_rotate_inv(q[par],total_moment[i]-wp.cross(p[par],total_force[i]))
        result[i]=torque+(dt*damping[i]+dt*dt*stiffness[i])*vector[i]

@wp.kernel(enable_backward=False)
def cg_update(rz:wp.array(dtype=float),pap:wp.array(dtype=float),direction:wp.array(dtype=wp.vec3),
              ap:wp.array(dtype=wp.vec3),inertia:wp.array(dtype=float),stiffness:wp.array(dtype=float),
              damping:wp.array(dtype=float),dt:float,omega:wp.array(dtype=wp.vec3),
              residual:wp.array(dtype=wp.vec3),z:wp.array(dtype=wp.vec3)):
    i=wp.tid()
    if i>0:
        alpha=float(0.0)
        if rz[0]>1.e-18 and pap[0]>1.e-20:alpha=rz[0]/pap[0]
        omega[i]+=alpha*direction[i];residual[i]-=alpha*ap[i]
        z[i]=residual[i]/(inertia[i]+dt*damping[i]+dt*dt*stiffness[i])

@wp.kernel(enable_backward=False)
def cg_direction(old:wp.array(dtype=float),new:wp.array(dtype=float),z:wp.array(dtype=wp.vec3),direction:wp.array(dtype=wp.vec3)):
    i=wp.tid();beta=float(0.0)
    if old[0]>1.e-18:beta=new[0]/old[0]
    direction[i]=z[i]+beta*direction[i]

@wp.kernel(enable_backward=False)
def integrate_angles(theta:wp.array(dtype=wp.vec3),omega:wp.array(dtype=wp.vec3),dt:float):
    i=wp.tid();theta[i]+=dt*omega[i]

@wp.kernel(enable_backward=False)
def leaf_integrate(torque:wp.array(dtype=float),air_damping:wp.array(dtype=float),inertia:wp.array(dtype=float),stiffness:wp.array(dtype=float),
                   damping:wp.array(dtype=float),theta:wp.array(dtype=float),omega:wp.array(dtype=float),dt:float):
    i=wp.tid()
    # Petiole stiffening and the drag Jacobian are integrated implicitly.
    elastic=stiffness[i]*theta[i]*(1.0+0.8*theta[i]*theta[i])
    tangent=stiffness[i]*(1.0+2.4*theta[i]*theta[i])
    w=(inertia[i]*omega[i]+dt*(torque[i]+air_damping[i]*omega[i]-elastic))/(inertia[i]+dt*(damping[i]+air_damping[i])+dt*dt*tangent)
    theta[i]+=dt*w;omega[i]=w

class Tree:
    def __init__(self,asset,device='cuda:0',iterations=40):
        self.asset=asset;self.device=device;self.n=len(asset['parent']);self.nl=len(asset['leaf_owner'])
        for name in ['parent','leaf_owner']:
            setattr(self,name,wp.array(asset[name],dtype=int,device=device))
        for name in ['position','edge','leaf_direction','leaf_across','leaf_normal']:
            setattr(self,name,wp.array(asset[name],dtype=wp.vec3,device=device))
        for name in ['mass','radius','length','inertia','stiffness','damping','leaf_size','leaf_area','leaf_inertia','leaf_stiffness','leaf_damping']:
            setattr(self,name,wp.array(asset[name],dtype=float,device=device))
        self.base=wp.vec3(*map(float,asset['position'][0]))
        self.p=wp.clone(self.position);self.v=wp.zeros(self.n,dtype=wp.vec3,device=device)
        self.q=wp.array(np.tile([0,0,0,1],(self.n,1)),dtype=wp.quat,device=device)
        self.theta=wp.zeros(self.n,dtype=wp.vec3,device=device);self.omega=wp.zeros_like(self.theta)
        self.force=wp.zeros_like(self.theta);self.moment=wp.zeros_like(self.theta)
        self.total_force=wp.zeros_like(self.theta);self.total_moment=wp.zeros_like(self.theta)
        self.leaf_theta=wp.zeros(self.nl,dtype=float,device=device);self.leaf_omega=wp.zeros_like(self.leaf_theta)
        self.leaf_torque=wp.zeros_like(self.leaf_theta)
        self.leaf_air_damping=wp.zeros_like(self.leaf_theta)
        self.iterations=iterations
        for name in ['residual','z','search','ap','mf','mm','mf_total','mm_total']:
            setattr(self,name,wp.zeros_like(self.theta))
        self.rz=wp.zeros(1,dtype=float,device=device);self.rz_new=wp.zeros_like(self.rz);self.pap=wp.zeros_like(self.rz)
        self.initial_rz=wp.zeros_like(self.rz)

    def clear(self):
        for a in [self.force,self.moment,self.total_force,self.total_moment,self.leaf_torque,self.leaf_air_damping]:a.zero_()

    def advance(self,dt):
        wp.launch(accumulate_load,self.n,[self.parent,self.force,self.moment,self.total_force,self.total_moment],device=self.device)
        wp.launch(cg_initialize,self.n,[self.parent,self.p,self.q,self.total_force,self.total_moment,self.inertia,
                  self.stiffness,self.damping,self.theta,self.omega,dt,self.residual,self.z,self.search],device=self.device)
        wp.utils.array_inner(self.residual,self.z,out=self.rz);wp.copy(self.initial_rz,self.rz)
        for _ in range(self.iterations):
            self.mf_total.zero_();self.mm_total.zero_()
            wp.launch(mass_action,self.n,[self.parent,self.p,self.q,self.mass,self.search,self.mf,self.mm],device=self.device)
            wp.launch(accumulate_load,self.n,[self.parent,self.mf,self.mm,self.mf_total,self.mm_total],device=self.device)
            wp.launch(matrix_finish,self.n,[self.parent,self.p,self.q,self.mf_total,self.mm_total,self.stiffness,self.damping,dt,self.search,self.ap],device=self.device)
            wp.utils.array_inner(self.search,self.ap,out=self.pap)
            wp.launch(cg_update,self.n,[self.rz,self.pap,self.search,self.ap,self.inertia,self.stiffness,self.damping,dt,
                      self.omega,self.residual,self.z],device=self.device)
            wp.utils.array_inner(self.residual,self.z,out=self.rz_new)
            wp.launch(cg_direction,self.n,[self.rz,self.rz_new,self.z,self.search],device=self.device);wp.copy(self.rz,self.rz_new)
        wp.launch(integrate_angles,self.n,[self.theta,self.omega,dt],device=self.device)
        wp.launch(leaf_integrate,self.nl,[self.leaf_torque,self.leaf_air_damping,self.leaf_inertia,self.leaf_stiffness,self.leaf_damping,self.leaf_theta,self.leaf_omega,dt],device=self.device)
        wp.launch(global_rotations,self.n,[self.theta,self.parent,self.q],device=self.device)
        wp.launch(forward,self.n,[self.parent,self.edge,self.q,self.p,self.v,self.base,dt],device=self.device)

    def state(self):
        return dict(p=self.p.numpy(),q=self.q.numpy(),v=self.v.numpy(),theta=self.theta.numpy(),
                    omega=self.omega.numpy(),leaf_theta=self.leaf_theta.numpy(),leaf_omega=self.leaf_omega.numpy())

    def diagnostics(self):
        state=self.state();p=state['p'];parent=self.asset['parent']
        lengths=np.linalg.norm(p[1:]-p[parent[1:]],axis=1)
        energy=.5*np.sum(self.asset['mass'][:,None]*state['v']**2+self.asset['stiffness'][:,None]*state['theta']**2)
        energy+=float(np.sum(.5*self.asset['leaf_inertia']*state['leaf_omega']**2+
                             self.asset['leaf_stiffness']*(.5*state['leaf_theta']**2+.2*state['leaf_theta']**4)))
        return dict(max_segment_length_error=float(np.max(np.abs(lengths-self.asset['length'][1:]))),
                    base_error=float(np.linalg.norm(p[0]-self.asset['position'][0])),
                    max_displacement=float(np.linalg.norm(p-self.asset['position'],axis=1).max()),
                    max_joint_angle=float(np.linalg.norm(state['theta'],axis=1).max()),
                    max_leaf_angle=float(np.max(np.abs(state['leaf_theta']))),elastic_kinetic_energy=float(energy),
                    relative_solver_residual=float(np.sqrt(max(0,self.rz.numpy()[0])/max(1.e-18,self.initial_rz.numpy()[0]))))
