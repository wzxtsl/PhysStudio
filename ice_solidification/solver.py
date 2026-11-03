"""Growing, boundary-attached ice coupled to position-based liquid motion."""
import numpy as np
import warp as wp
from ice_melting import solver as fluid
from ice_melting import thermal as material
from .thermal import ImplicitHeat
from . import capillary
from .boundaries import project_star

wp.set_module_options({'enable_backward':False})


@wp.kernel
def phase_update(x:wp.array(dtype=wp.vec3),e:wp.array(dtype=float),groups:wp.array(dtype=int),
                 anchors:wp.array(dtype=wp.vec3),birth:wp.array(dtype=float),time:float):
    i=wp.tid()
    if e[i]<=0. and groups[i]<0:
        groups[i]=0;anchors[i]=x[i];birth[i]=time
    elif e[i]>=333500. and groups[i]>=0:
        groups[i]=-1;birth[i]=-1.


@wp.kernel
def pin(x:wp.array(dtype=wp.vec3),anchors:wp.array(dtype=wp.vec3),groups:wp.array(dtype=int)):
    i=wp.tid()
    if groups[i]>=0:x[i]=anchors[i]


@wp.kernel
def relax(delta:wp.array(dtype=wp.vec3),factor:float):
    i=wp.tid();delta[i]*=factor


@wp.kernel
def predict(x:wp.array(dtype=wp.vec3),v:wp.array(dtype=wp.vec3),old:wp.array(dtype=wp.vec3),
            groups:wp.array(dtype=int),e:wp.array(dtype=float),drag:float,dt:float):
    i=wp.tid();old[i]=x[i]
    if groups[i]>=0:
        v[i]=wp.vec3(0.)
    else:
        f=material.liquid(e[i]);resistance=drag*(1.-f)*(1.-f)/(f*f*f+.001)
        v[i]=(v[i]+wp.vec3(0.,0.,-9.81)*dt)/(1.+dt*resistance)
        x[i]+=dt*v[i]


@wp.kernel
def project_geometry(x:wp.array(dtype=wp.vec3),mode:int,bowl:float,rod:wp.vec3,rod_r:float,radius:float):
    i=wp.tid();p=x[i]
    if mode==1:
        length=wp.length(wp.vec3(p[0],p[1],0.))
        if length>bowl-radius:
            p=wp.vec3(p[0]*(bowl-radius)/length,p[1]*(bowl-radius)/length,p[2])
        delta=wp.vec3(p[0]-rod[0],p[1]-rod[1],0.);length=wp.length(delta)
        if length<rod_r+radius:
            normal=wp.vec3(1.,0.,0.)
            if length>1.e-8:normal=delta/length
            p+=(rod_r+radius-length)*normal
    elif mode==2:
        if p[0]<-.025+radius and p[2]<.026+radius:
            if .026+radius-p[2]<-.025+radius-p[0]:p=wp.vec3(p[0],p[1],.026+radius)
            else:p=wp.vec3(-.025+radius,p[1],p[2])
        if p[0]<.025+radius and p[2]<.014+radius:
            if .014+radius-p[2]<.025+radius-p[0]:p=wp.vec3(p[0],p[1],.014+radius)
            else:p=wp.vec3(.025+radius,p[1],p[2])
    x[i]=p


class FreezingSystem(fluid.IceSystem):
    def __init__(self,x,v,g,e,cfg,device='cuda:0'):
        super().__init__(x,g,e,cfg,device)
        self.v.assign(v);self.anchors=wp.clone(self.x)
        self.birth=wp.full(self.n,-1.,dtype=float,device=device)
        self.color_normal=wp.zeros_like(self.x)
        self.heat_solver=ImplicitHeat(self)
        self.max_density=0.

    def shape_match(self):
        self.launch(pin,self.n,[self.x,self.anchors,self.groups])

    def update_components(self):
        self.launch(phase_update,self.n,[self.x,self.e,self.groups,self.anchors,self.birth,self.time])

    def heat(self,dt):
        cfg=self.cfg;t=self.time+.5*dt
        ramp=np.clip((t-cfg.get('cool_start',0.))/cfg.get('cool_ramp',.3),0.,1.)
        ramp=ramp*ramp*(3-2*ramp)
        cold=cfg['initial_t']+(cfg['cold_t']-cfg['initial_t'])*ramp
        self.heat_solver.advance(dt*cfg['thermal_scale'],cold)
        self.update_components()

    def step(self,dt=1/720,iterations=6,heating=False):
        density_kernel=capillary.plate_density if self.cfg['boundary_mode']==0 else fluid.density
        self.grid.build(self.x,self.h)
        self.launch(density_kernel,self.n,[self.x,self.e,self.groups,self.grid.id,self.h,self.norm,self.lam,self.rho])
        self.launch(fluid.normals,self.n,[self.x,self.e,self.groups,self.grid.id,self.h,self.norm,self.normal])
        self.launch(fluid.surface_forces,self.n,[self.x,self.v,self.vout,self.e,self.groups,self.normal,self.rho,
            self.grid.id,self.h,self.norm,self.mass,dt,self.cfg.get('surface_tension',.025),
            self.cfg.get('adhesion',.035),self.cfg.get('ice_drag',.25),self.cfg.get('viscosity',8.)])
        self.v,self.vout=self.vout,self.v
        if self.cfg.get('csf_sigma',0.):
            mirror=int(self.cfg['boundary_mode']==0)
            self.launch(capillary.normals,self.n,[self.x,self.e,self.grid.id,self.h,self.norm,mirror,self.color_normal])
            self.launch(capillary.accelerate,self.n,[self.x,self.v,self.e,self.groups,self.color_normal,
                self.grid.id,self.h,self.norm,self.cfg['csf_sigma'],dt,mirror])
        if self.cfg.get('wall_drag',0.):
            self.launch(fluid.wall_film_viscosity,self.n,[self.x,self.v,self.e,self.groups,0.,self.dx*.46,
                self.dx*self.cfg.get('wall_depth_dx',2.),self.cfg['wall_drag'],dt])
        self.launch(predict,self.n,[self.x,self.v,self.old,self.groups,self.e,self.cfg.get('porosity_drag',25.),dt])
        for _ in range(iterations):
            self.grid.build(self.x,self.h)
            self.launch(density_kernel,self.n,[self.x,self.e,self.groups,self.grid.id,self.h,self.norm,self.lam,self.rho])
            self.launch(fluid.corrections,self.n,[self.x,self.e,self.groups,self.grid.id,self.h,self.norm,self.dx,self.lam,self.delta,self.limited])
            if self.cfg['boundary_mode']==0:
                self.launch(capillary.plate_corrections,self.n,[self.x,self.e,self.groups,self.grid.id,self.h,self.norm,self.dx,self.lam,self.delta])
            if self.cfg.get('projection_relaxation',1.)!=1.:
                self.launch(relax,self.n,[self.delta,self.cfg['projection_relaxation']])
            self.launch(fluid.apply,self.n,[self.x,self.delta,self.lo,self.hi,self.dx*.46])
            self.launch(project_geometry,self.n,[self.x,self.cfg['boundary_mode'],self.cfg.get('bowl_radius',0.),
                wp.vec3(*self.cfg.get('rod_center',[0.,0.,0.])),self.cfg.get('rod_radius',0.),self.dx*.46])
            if self.cfg['boundary_mode']==3:
                self.launch(project_star,self.n,[self.x,self.cfg['star_radius'],self.dx*.46])
            self.shape_match()
        self.launch(fluid.velocities,self.n,[self.x,self.old,self.v,dt,self.lo,self.hi,self.dx*.46])
        self.time+=dt;self.steps+=1

    def snapshot(self):
        data=super().snapshot();data['freeze_time']=self.birth.numpy()
        return data

    def audit(self):
        data=self.snapshot()
        for name,value in data.items():
            if not np.isfinite(value).all():raise RuntimeError(f'Nonfinite {name}')
        e=data['enthalpy'];f=material.fraction(e);v=data['velocities'];g=data['groups'];x=data['positions']
        energy=float(e.astype(float).sum()*self.mass)
        external=self.heat_solver.external_energy
        speed=np.linalg.norm(v,axis=1);fixed=g>=0
        return dict(time=self.time,particles=self.n,mass_kg=self.mass*self.n,energy_J=energy,
            external_heat_J=external,energy_balance_J=energy-self.initial_energy-external,
            relative_energy_balance=(energy-self.initial_energy-external)/(self.mass*self.n*material.LATENT),
            ice_fraction=float(1-f.mean()),frozen_particles=int(fixed.sum()),
            physical_volume=float(np.sum(self.mass*((1-f)/material.RHO_ICE+f/material.RHO_WATER))),
            min_temperature=float(material.temperature(e).min()),max_temperature=float(material.temperature(e).max()),
            max_speed=float(speed.max()),rms_liquid_speed=float(np.sqrt(np.mean(speed[~fixed]**2))) if (~fixed).any() else 0.,
            fixed_max_speed=float(speed[fixed].max()) if fixed.any() else 0.,
            implicit_iterations=self.heat_solver.last_iterations,max_implicit_iterations=self.heat_solver.maximum_iterations,
            maximum_thermal_residual_J=self.heat_solver.maximum_residual,
            density_p95=float(np.quantile(data['density'][~fixed],.95)) if (~fixed).any() else 0.)
