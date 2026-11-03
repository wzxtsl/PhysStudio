"""Independent physical identities and strong-contact acceptance checks."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from .scenes import base,body,pair_test
from .solver import Solver,rotate


def derivative_check(device):
    cfg=base('derivatives');cfg.update(gravity=[0,0,0],acceleration_tolerance=.01)
    cfg['bodies'] += [body('a',[-.13,.04,.12],r=.025,mass=.1,moment=[2,1,-.4]),
                      body('b',[.16,-.02,.08],r=.025,mass=.2,moment=[-.3,.8,1.5])]
    cfg['bodies'][1]['samples'][0]['p']=[.01,-.005,0]
    cfg['bodies'][2]['samples'][0]['p']=[-.008,.004,.002]
    s=Solver(cfg,device);dt=.008;s.prepare(dt,None)
    u=np.zeros(s.n*6);E,g=s.evaluate(u,dt);H=s.hessian(u,dt)
    r=s.x[1]+s.sl[0]-s.x[2]-s.sl[1];d=np.linalg.norm(r);n=r/d
    a,b=s.m
    F=3.e-7/d**4*((a@n)*b+(b@n)*a+(a@b)*n-5*(a@n)*(b@n)*n)
    B1=1.e-7/d**3*(3*n*(b@n)-b)
    B2=1.e-7/d**3*(3*n*(a@n)-a)
    ta=np.cross(a,B1)+np.cross(s.sl[0],F)
    tb=np.cross(b,B2)+np.cross(s.sl[1],-F)
    expected=np.zeros_like(g)
    expected[6:9]=-dt*s.sv[1]*F;expected[9:12]=-dt*s.sw[1]*ta
    expected[12:15]=dt*s.sv[2]*F;expected[15:18]=-dt*s.sw[2]*tb
    gradient_error=float(np.max(np.abs(g-expected)))
    assert np.allclose(g,expected,rtol=1.e-8,atol=1.e-11),(g,expected)
    jac=np.zeros_like(H)
    for j in s.active:
        du=np.zeros_like(u);du[j]=1.e-5
        jac[:,j]=(s.evaluate(u+du,dt)[1]-s.evaluate(u-du,dt)[1])/(2.e-5)
    error=float(np.max(np.abs(H[np.ix_(s.active,s.active)]-jac[np.ix_(s.active,s.active)])))
    assert error<3.e-5,error
    far_cfg=copy.deepcopy(cfg);far_cfg['distance_threshold']=0.
    far=Solver(far_cfg,device);far.prepare(dt,None)
    gf=far.evaluate(u,dt)[1];hf=far.hessian(u,dt)
    assert np.max(np.abs(gf-g))<1.e-10
    assert np.max(np.abs(hf-far.inertia_h))<1.e-10
    return dict(analytic_force_torque_gradient_error=gradient_error,hessian_vs_gradient_jacobian_error=error,
                hybrid_gradient_continuous=True,far_hessian_zero=True)


def induction_check(device):
    cfg=base('induction');cfg.update(gravity=[0,0,0])
    cfg['bodies'] += [body('permanent',[0,0,0],r=.04,moment=[0,0,20],free=[0]*6),
                      body('linear',[0,0,.3],r=.03,chi=99)]
    s=Solver(cfg,device);s.magnetize()
    volume=4*np.pi/3*(.03*.72)**3
    H=2*20/(4*np.pi*.3**3)
    expected=volume*99/(1+99/3)*H
    assert np.allclose(s.m[1],[0,0,expected],rtol=1.e-12,atol=1.e-12)
    return dict(expected_moment=expected,warp_moment=s.m[1].tolist())


def contact_check(device,moment,dt):
    cfg=pair_test(moment);cfg['acceleration_tolerance']=.08
    # Magnetic attraction scales as moment squared. Scale the contact energy
    # with it to preserve a comparable contact equilibrium thickness.
    cfg['contact_stiffness']=1.e8*(moment/50.)**2
    cfg['bodies'][1]['damping']=cfg['bodies'][2]['damping']=0.
    s=Solver(cfg,device);positions=[]
    for _ in range(80):
        s.step(dt);positions.append(s.x.copy())
    gaps=[h['min_gap'] for h in s.history]
    separation=np.array([p[2,0]-p[1,0] for p in positions])
    imbalance=float(np.linalg.norm(s.mass@s.v))
    assert min(gaps)>0
    assert separation[-1]<.085, separation[-1]
    assert imbalance<1.e-8,imbalance
    assert np.max(np.abs(s.q[:,:3]))<1.e-10
    return dict(moment=moment,dt=dt,accepted_steps=len(s.history),min_gap=min(gaps),final_separation=separation[-1],
                final_speed=float(np.max(np.abs(s.v))),momentum_error=imbalance,
                max_retry_depth=max(h['retry_depth'] for h in s.history),final_kappa=s.kappa,
                contact_tail_range=float(np.ptp(separation[-20:])))


def friction_check(device):
    cfg=base('friction');cfg.update(planes=[dict(normal=[0,0,1],offset=0)],friction=.3,
                                   acceleration_tolerance=.03)
    mass=.1;r=.04;gap=cfg['contact_thickness']-np.sqrt(mass*9.81/(3*cfg['contact_stiffness']))
    ball=body('rolling sphere',[0,0,r+gap],r=r,mass=mass)
    ball['samples']=[];ball.update(damping=0,angular_damping=0)
    cfg['bodies'].append(ball);s=Solver(cfg,device);s.v[1,0]=.25
    initial=.5*mass*.25**2
    for _ in range(45):s.step(.005)
    energy=.5*mass*np.dot(s.v[1],s.v[1])+.5*np.dot(s.I[1],s.w[1]**2)
    assert energy<initial
    assert s.v[1,0]>.05 and s.w[1,1]>1
    assert min(h['min_gap'] for h in s.history)>0
    return dict(initial_kinetic=initial,final_kinetic=energy,velocity=s.v[1].tolist(),omega=s.w[1].tolist())


def swept_collision_check(device):
    cfg=pair_test(0.);cfg['acceleration_tolerance']=.1
    s=Solver(cfg,device);s.v[1,0]=20.;s.v[2,0]=-20.
    initial=float(.5*np.sum(s.mass[:,None]*s.v*s.v))
    for _ in range(8):s.step(.02)
    assert s.x[1,0]<s.x[2,0], 'Fast spheres crossed through one another'
    assert min(h['min_gap'] for h in s.history)>0
    final=float(.5*np.sum(s.mass[:,None]*s.v*s.v))
    assert final<initial
    return dict(initial_speed=20.,dt=.02,min_gap=min(h['min_gap'] for h in s.history),
                initial_kinetic=initial,final_kinetic=final,accepted_steps=len(s.history))


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report={}
    checks=[('derivatives',lambda:derivative_check(a.device)),('induction',lambda:induction_check(a.device)),
            ('contact',lambda:contact_check(a.device,50.,.01)),('strong_contact',lambda:contact_check(a.device,500.,.002)),
            ('friction',lambda:friction_check(a.device)),('swept_collision',lambda:swept_collision_check(a.device))]
    for name,fn in checks:
        report[name]=fn();print(name,json.dumps(report[name]),flush=True)
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    report['passed']=True;a.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
