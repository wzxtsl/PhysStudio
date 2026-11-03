"""Procedural cooled boundaries for three freezing demonstrations."""
import copy
import numpy as np
from ice_melting.scenes import lattice
from ice_melting.thermal import LATENT,CP_WATER,RHO_ICE,RHO_WATER
from .boundaries import distance_and_reflection

PRESETS = {
 'ice_cast':dict(label='Frozen star',dx=.0015,seconds=3.5,frames=241,thermal_scale=330.,
    bounds=[[-.038,-.038,0.],[.038,.038,.06]],target=[0.,0.,.012],boundary_mode=3,
    star_radius=.033,water_height=.017,render_scale=65.,
    cold_t=-28.,contact_h=1200.,air_t=2.,air_h=2.,initial_t=2.,cool_start=.10,cool_ramp=1.40,
    thermal_substeps=4,phase_rendering='continuous',
    viscosity=15.,surface_tension=.008,porosity_drag=25.,camera_start=-125.,camera_orbit=70.),
 'cold_finger':dict(label='Cryogenic bloom',dx=.0023,seconds=5.,frames=181,thermal_scale=1100.,
    bounds=[[-.057,-.057,0.],[.057,.057,.095]],target=[0.,0.,.027],boundary_mode=1,
    cylindrical=True,bowl_radius=.055,water_height=.036,rod_center=[-.010,0.,0.],rod_radius=.0065,
    cold_t=-38.,contact_h=5000.,air_t=2.,air_h=3.,initial_t=2.,cool_start=.35,cool_ramp=.35,
    viscosity=8.,surface_tension=.025,porosity_drag=25.,camera_start=-125.,camera_orbit=65.),
 'cascade':dict(label='Frozen cascade',dx=.0022,seconds=4.2,frames=241,thermal_scale=1000.,
    bounds=[[-.090,-.032,0.],[.090,.032,.115]],target=[0.,0.,.029],boundary_mode=2,
    cold_t=-32.,contact_h=650.,air_t=2.,air_h=2.,initial_t=3.,cool_start=.20,cool_ramp=2.50,
    thermal_substeps=4,phase_rendering='continuous',
    viscosity=60.,surface_tension=0.,adhesion=0.,ice_drag=0.,porosity_drag=1000.,
    projection_relaxation=.25,mechanical_dt=1/1440,projection_iterations=12,
    camera_start=-120.,camera_orbit=65.),

}


def build(name, dx=None, settings=None):
    cfg=copy.deepcopy(PRESETS[name]);cfg['scene']=name
    if settings:cfg.update(settings)
    if dx is not None:cfg['dx']=dx
    dx=cfg['dx'];spacing=dx*(RHO_ICE/RHO_WATER)**(1/3)
    if name=='ice_cast':
        radius=cfg['star_radius'];height=cfg['water_height']
        x=lattice([-radius,-radius,0.],[radius,radius,height],spacing)
        distance,_=distance_and_reflection(x,radius);x=x[distance<-.5*dx]
        v=np.zeros_like(x)
    elif name=='cold_finger':
        radius=cfg['bowl_radius'];height=cfg['water_height']
        x=lattice([-radius,-radius,0.],[radius,radius,height],spacing)
        r=np.linalg.norm(x[:,:2],axis=1);rod=np.array(cfg['rod_center'])
        x=x[(r<radius-dx*.5)&(np.linalg.norm((x-rod)[:,:2],axis=1)>cfg['rod_radius']+.55*dx)]
        v=np.c_[-x[:,1],x[:,0],np.zeros(len(x))]*1.8
    elif name=='cascade':
        x=lattice([-.084,-.026,.026],[.002,.026,.060],spacing)
        floor=np.where(x[:,0]<-.025,.026,.014)
        x=x[x[:,2]>floor+.45*dx]
        v=np.zeros_like(x);v[:,0]=.10
    else:
        raise ValueError(f'Unknown scene: {name}')
    e=np.full(len(x),LATENT+cfg['initial_t']*CP_WATER,np.float32)
    g=np.full(len(x),-1,np.int32)
    cfg.update(particles=len(x),mass=RHO_ICE*dx**3,playback_fps=30)
    return x.astype(np.float32),v.astype(np.float32),g,e,cfg
