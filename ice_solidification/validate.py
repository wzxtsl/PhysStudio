"""Conservation, latent heat, and a one-dimensional Stefan freezing check."""
import argparse,json,math
from pathlib import Path
import numpy as np
import warp as wp
from scipy.optimize import brentq
from scipy.special import erf
from ice_melting import thermal as material
from ice_melting.scenes import lattice
from .thermal import ImplicitHeat,jacobi


class ThermalCloud:
    def __init__(self,x,e,dx,cfg,device):
        self.n=len(x);self.dx=dx;self.h=2.3*dx;self.mass=917*dx**3;self.device=device;self.cfg=cfg
        offsets=lattice([-3.5*dx]*3,[3.5*dx]*3,dx)
        self.norm=float(np.sum(np.maximum(1.-np.sum(offsets**2,axis=1)/self.h**2,0.)**3))
        self.x=wp.array(x,dtype=wp.vec3,device=device);self.e=wp.array(e,dtype=float,device=device)
        self.grid=wp.HashGrid(64,64,64,device=device)
        self.heat=ImplicitHeat(self)
    def launch(self,k,n,args):wp.launch(k,n,inputs=args,device=self.device)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--skip-stefan',action='store_true');args=p.parse_args()
    wp.init();device=args.device;result={}
    # A constant sink removes latent heat without lowering temperature prematurely.
    s=ThermalCloud(np.array([[0,0,.01]],np.float32),np.array([material.LATENT],np.float32),.002,
        dict(boundary_mode=0,contact_h=0.),device)
    h=s.heat;h.bb.fill_(-s.mass*material.LATENT)
    enthalpies=[]
    for _ in range(3):
        wp.copy(h.old,s.e)
        s.launch(jacobi,1,[h.old,s.e,h.scratch,h.ids,h.weights,h.count,h.diagonal,h.bd,h.bb,.5,s.mass])
        s.e,h.scratch=h.scratch,s.e;enthalpies.append(float(s.e.numpy()[0]))
    assert np.allclose(enthalpies,[material.LATENT/2,0.,-material.LATENT/2],atol=.03)
    result['latent_plateau']=dict(enthalpies=enthalpies,temperatures=material.temperature(enthalpies).tolist(),passed=True)
    x=np.array([[0,0,.01],[.0015,0,.01]],np.float32)
    e=np.array([material.LATENT+4180*20,material.LATENT+4180*5],np.float32)
    s=ThermalCloud(x,e,.002,dict(boundary_mode=0,contact_h=0.),device)
    before=float(e.sum(dtype=float)*s.mass);s.heat.advance(50.,0.)
    after=float(s.e.numpy().sum(dtype=float)*s.mass)
    assert abs(after-before)/(s.mass*2*material.LATENT)<1.e-6
    t=material.temperature(s.e.numpy());assert 5.<=t.min()<=t.max()<=20.
    result['closed_heat_exchange']=dict(energy_error_J=after-before,temperatures=t.tolist(),passed=True)
    x=lattice([0,0,0],[.012,.012,.012],.0015)
    e=np.full(len(x),material.LATENT+4180*2,np.float32)
    s=ThermalCloud(x,e,.0015,dict(boundary_mode=0,contact_h=1800.,air_t=2.,air_h=2.),device)
    for _ in range(8):s.heat.advance(4.95,2.)
    drift=float(np.max(np.abs(s.e.numpy()-e)))
    assert drift<=.03125,drift
    result['isothermal_fixed_point']=dict(max_enthalpy_drift_J_kg=drift,passed=True)
    if not args.skip_stefan:
        dx=.0015;nx=12;ny=12;nz=40
        x=lattice([0,0,0],[nx*dx,ny*dx,nz*dx],dx)
        e=np.full(len(x),material.LATENT,np.float32)
        cfg=dict(boundary_mode=0,contact_h=20000.,air_h=0.)
        s=ThermalCloud(x,e,dx,cfg,device);rows=[]
        alpha=2.2/(917*2100);ste=2100*10/material.LATENT
        lam=brentq(lambda z:math.sqrt(math.pi)*z*math.exp(z*z)*erf(z)-ste,.001,2.)
        for step in range(80):
            s.heat.advance(20.,-10.)
            if (step+1)%20==0:
                time=(step+1)*20.
                front=float((1-material.fraction(s.e.numpy())).sum()/(nx*ny)*dx)
                exact=2*lam*math.sqrt(alpha*time)
                rows.append(dict(time_s=time,front_m=front,analytic_m=exact,relative_error=abs(front-exact)/exact))
        energy=float(s.e.numpy().sum(dtype=float)*s.mass)
        ledger=energy-float(e.sum(dtype=float)*s.mass)-s.heat.external_energy
        result['stefan']=dict(samples=rows,energy_balance_J=ledger,maximum_iterations=s.heat.maximum_iterations)
        assert rows[-1]['relative_error']<.20,rows
    result['passed']=True;args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
