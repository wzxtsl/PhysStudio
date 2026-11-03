"""Procedural physical scenes; moving actuators are explicit inputs."""
import copy
import math
import numpy as np


def sphere(p,r): return {'p':list(p),'r':r}


def body(name,pos,r=.035,mass=.4,moment=None,chi=0,free=None,shape=None,color='steel'):
    shape=shape or [sphere([0,0,0],r)]
    inertia=np.zeros(3)
    for s in shape:
        p=np.array(s['p']); inertia+=mass/len(shape)*(.4*s['r']**2+np.dot(p,p)-p*p)
    b=dict(name=name,pos=list(pos),mass=mass,inertia=inertia.tolist(),shape=shape,color=color,
           samples=[dict(p=s['p'],r=s['r']*.72,moment=(np.array(moment or [0,0,0])/len(shape)).tolist(),chi=chi) for s in shape])
    if free is not None:b['free']=free
    return b


def base(name):
    return dict(scene=name,fps=30,duration=10.,substeps=4,gravity=[0,0,-9.81],contact_thickness=.0015,
                contact_stiffness=1.e8,friction=.22,friction_velocity=.01,acceleration_tolerance=.2,
                newton_iterations=40,hessian_epsilon=2.e-4,
                bodies=[dict(name='world',pos=[0,0,0],mass=1.,inertia=[1,1,1],shape=[],samples=[],free=[0]*6)])


def make(name):
    c=base(name)
    if name=='compass_garden':
        c.update(title='Field of needles',duration=10.,distance_threshold=.36,gravity=[0,0,0],friction=0)
        source=body('Rotating bar magnet',[0,0,.10],mass=5.,moment=[240,0,0],free=[0]*6,
                    shape=[sphere([x,0,0],.045) for x in np.linspace(-.12,.12,7)],color='magnet')
        c['bodies'].append(source)
        for y in np.linspace(-.59,.59,8):
            for x in np.linspace(-.72,.72,10):
                if x*x+y*y<.235**2:continue
                needle=body('Compass', [x,y,.075],mass=.012,moment=[.35,0,0],free=[0,0,0,0,0,1],
                            shape=[sphere([xx,0,0],.012) for xx in [-.035,0,.035]],color='compass')
                angle=math.pi*.5
                needle['quat']=[0,0,math.sin(angle/2),math.cos(angle/2)]
                needle['angular_damping']=1.8
                needle['samples']=[dict(p=[0,0,0],r=.02,moment=[.35,0,0],chi=0)]
                c['bodies'].append(needle)
    elif name=='magnetic_crane':
        c.update(title='Electromagnetic foundry',duration=12.,distance_threshold=.60,magnetic_strength=6000.,
                 planes=[dict(normal=[0,0,1],offset=0),
                         dict(normal=[1,0,0],offset=-1.03),dict(normal=[-1,0,0],offset=-1.03),
                         dict(normal=[0,1,0],offset=-.70),dict(normal=[0,-1,0],offset=-.70)])
        c['bodies'].append(dict(name='Gantry trolley',pos=[-.40,0,.90],mass=1.,inertia=[1]*3,shape=[],samples=[],free=[0]*6))
        magnet=body('Suspended electromagnet',[-.40,0,.40],r=.115,mass=7.,moment=[0,0,0],color='coil')
        magnet['angular_damping']=.5
        c['bodies'].append(magnet)
        c['joints']=[dict(kind=5,a=2,b=1,ra=[0,0,.10],length=.40,stiffness=1.e7)]
        for iy in range(5):
            for ix in range(5):
                c['bodies'].append(body('Nickel bead',[-.40+(ix-2)*.074,(iy-2)*.074,.037],mass=.48,chi=99,
                                        color='steel' if (ix+iy)%3 else 'brass'))
        for ix,iy in [(1,1),(2,1),(3,1),(1,2),(2,2),(3,2),(2,3)]:
            c['bodies'].append(body('Nickel bead',[-.40+(ix-2)*.074+.02,(iy-2)*.074+.02,.112],mass=.48,chi=99,color='steel'))
        c['coil_body']=2
    elif name=='magnetic_globe':
        c.update(title='Magnetic cabinet',duration=12.,distance_threshold=.45,friction=.16,magnetic_strength=8000.,
                 container=dict(center=[0,0,.47],radius=.405))
        magnet=body('Orbiting electromagnet',[0,0,-.08],r=.105,mass=5.,moment=[0,0,0],free=[0]*6,color='coil')
        magnet['inside']=False;c['bodies'].append(magnet)
        for layer in range(2):
            for y in [-.125,0,.125]:
                for x in [-.15,0,.15]:
                    if (layer==1 and x==0 and y==0): continue
                    # Rounded rigid triads; collision and visible geometry share spheres.
                    shape=[sphere([-.025,0,0],.029),sphere([.025,0,0],.029),sphere([0,.035,0],.024)]
                    obj=body('Nickel triad',[x,y,.205+layer*.105],mass=.26,chi=99,shape=shape,
                             color=['steel','brass','cobalt'][(len(c['bodies'])+layer)%3])
                    a=(len(c['bodies'])*2.39996)%6.28
                    obj['quat']=[0,0,math.sin(a/2),math.cos(a/2)]
                    c['bodies'].append(obj)
        c['coil_body']=1
    else:raise ValueError(name)
    return c


def smooth(a,b,t):
    s=np.clip(t,0,1);s=s*s*(3-2*s)
    return np.asarray(a)*(1-s)+np.asarray(b)*s


def control(s,t):
    name=s.cfg['scene']
    if name=='compass_garden':
        angle=.70*t+.30*math.sin(t*.9)
        s.base_q[1]=[0,0,math.sin(angle/2),math.cos(angle/2)]
    elif name=='magnetic_crane':
        if t<1.7:p=smooth([-.40,0,.90],[-.40,0,.70],t/1.7)
        elif t<3.7:p=smooth([-.40,0,.70],[-.40,0,1.18],(t-1.7)/2)
        elif t<7.2:p=smooth([-.40,0,1.18],[.43,0,1.18],(t-3.7)/3.5)
        elif t<8.5:p=smooth([.43,0,1.18],[.43,0,.83],(t-7.2)/1.3)
        else:p=smooth([.43,0,.83],[.43,0,1.13],(t-9)/2)
        s.base_x[1]=p
        strength=float(smooth(0,1,(t-.7)/.7))*float(smooth(1,0,(t-8.45)/.25))
        s.permanent[s.so==2]=[0,0,s.cfg['magnetic_strength']*strength]
    elif name=='magnetic_globe':
        angle=float(smooth(0,math.pi,(t-1.4)/6.5))
        r=.55
        p=np.array([r*math.sin(angle),.035*math.sin(2*angle),.47-r*math.cos(angle)])
        s.base_x[1]=p
        direction=np.array([0,0,.47])-p;direction/=np.linalg.norm(direction)
        # Quaternion from +z to the inward pole direction.
        axis=np.cross([0,0,1],direction);qw=1+direction[2]
        q=np.r_[axis,qw]
        if np.linalg.norm(q)<1e-8:q=np.array([1,0,0,0])
        s.base_q[1]=q/np.linalg.norm(q)
        strength=float(smooth(0,1,(t-.35)/.8))*float(smooth(1,0,(t-9.0)/.25))
        s.permanent[s.so==1]=[0,0,s.cfg['magnetic_strength']*strength]


def pair_test(moment=5.,dt=.01):
    c=base('pair_test');c.update(gravity=[0,0,0],contact_stiffness=1.e8,acceleration_tolerance=.01)
    for x in [-.10,.10]:
        c['bodies'].append(body('test magnet',[x,0,.1],r=.04,mass=.1,moment=[moment,0,0],free=[1,0,0,0,0,0]))
    return c
