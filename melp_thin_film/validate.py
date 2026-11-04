"""Numerical checks against independent algebra and analytic sphere identities."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix,diags
import warp as wp
from warp.optim.linear import cg
from .scenes import make
from .solver import Solver
from . import kernels as K

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    results={};rng=np.random.default_rng(87);cfg=make();cfg.update(rotation_speed=0.,vortex_speed=0.)
    s=Solver(cfg);r=cfg['bubbles'][0]['radius'];a=s.area.numpy();x=s.x.numpy();n=s.n.numpy()
    results['sphere_geometry']=dict(area_relative_error=abs(a.sum()/(4*np.pi*r*r)-1),
        volume_relative_error=abs(s.volume.numpy()[0]/(4*np.pi*r**3/3)-1),
        normal_max_error=float(np.linalg.norm(n-x/r,axis=1).max()),curvature_relative_rms=float(np.sqrt(np.mean((s.curvature.numpy()*r/2-1)**2))))
    assert results['sphere_geometry']['area_relative_error']<.03
    assert results['sphere_geometry']['curvature_relative_rms']<.04
    vel=rng.normal(size=(s.nl,3));affine=rng.normal(size=(s.nl,3,3))*3
    s.v.assign(vel);s.affine.assign(affine);s.transfer()
    expected=(s.vol.numpy()[:,None]*vel).sum(0);actual=s.momentum.numpy().sum(0)
    results['conservative_affine_scatter']=dict(volume_relative_error=abs(s.ev.numpy().sum()/s.initial_liquid-1),
        surfactant_relative_error=abs(s.ec.numpy().sum()/s.initial_soap-1),momentum_relative_error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)))
    assert max(results['conservative_affine_scatter'].values())<1e-11
    constant=np.array([.031,-.075,.124]);s.v.assign(np.tile(constant,(s.nl,1)));s.affine.zero_();s.transfer()
    s.launch(K.e_to_l,s.nl,s.x,s.p,s.ln,s.velocity,s.lnb,s.lnum,s.weights,s.mean,s.v,s.affine,1.)
    results['constant_velocity_roundtrip']=float(np.max(np.abs(s.v.numpy()-constant)))
    assert results['constant_velocity_roundtrip']<1e-12
    # Assemble G independently from saved neighbor coefficients.
    nb=s.enb.numpy();counts=s.enum.numpy();grad=s.gradient.numpy();row=[];col=[];value=[]
    for i in range(s.ne):
        for k in range(counts[i]):
            j=nb[i,k]
            for d in range(3):
                row.extend([3*i+d,3*i+d]);col.extend([j,i]);value.extend([grad[i,k,d],-grad[i,k,d]])
    G=coo_matrix((value,(row,col)),shape=(3*s.ne,s.ne)).tocsr()
    grad_error=np.linalg.norm((G@x).reshape(s.ne,3,3)-(np.eye(3)[None]-n[:,:,None]*n[:,None,:]),axis=(1,2))
    s.dt=cfg['dt'];s.launch(K.compatible_init,s.ne,s.gamma,s.area,s.base_t,s.diag_t,s.rhs_t)
    s.launch(K.compatible_rhs,s.ne,s.velocity,s.n,s.area,s.thickness,s.enb,s.enum,s.gradient,K.V(),s.dt,cfg['rt'],cfg['rho'],1.,s.diag_t,s.rhs_t)
    b=a/s.gamma.numpy();d=np.repeat(2*cfg['rt']/cfg['rho']*a/s.thickness.numpy(),3)
    A=diags(b)+s.dt**2*G.T@diags(d)@G
    q=rng.normal(size=s.ne);u=wp.array(q,dtype=K.D,device=s.device);z=wp.zeros_like(u)
    s.t_matvec(u,z,z,1.,0.);expected=A@q
    results['compatible_pressure']=dict(operator_relative_error=float(np.linalg.norm(z.numpy()-expected)/np.linalg.norm(expected)),
        symmetry_max_error=float(abs(A-A.T).max()),coordinate_gradient_max_error=float(grad_error.max()),quadratic_energy=float(q@(A@q)))
    assert results['compatible_pressure']['operator_relative_error']<1e-12
    assert results['compatible_pressure']['symmetry_max_error']<1e-10
    assert results['compatible_pressure']['coordinate_gradient_max_error']<.02
    assert results['compatible_pressure']['quadratic_energy']>0
    s.launch(K.invert_diagonal,s.ne,s.diag_t,s.inv_t)
    target=rng.normal(size=s.ne);rhs=wp.array(A@target,dtype=K.D,device=s.device);sol=wp.zeros_like(rhs)
    it,err,tol=cg(s.A_t,rhs,sol,M=s.inv_t,tol=1e-9,atol=1e-14,maxiter=400)
    results['manufactured_cg']=dict(relative_solution_error=float(np.linalg.norm(sol.numpy()-target)/np.linalg.norm(target)),iterations=it,residual=err,tolerance=tol)
    assert results['manufactured_cg']['relative_solution_error']<1e-6
    # A sampled sphere has a small nonuniform surfactant quadrature error.
    # Check bounded shape drift, not an exact zero-velocity equilibrium.
    s=Solver(cfg)
    for _ in range(400):s.step(cfg['dt'])
    s.refresh();d=s.diagnostics()
    d['radial_relative_rms']=float(np.sqrt(np.mean((np.linalg.norm(s.x.numpy(),axis=1)/r-1)**2)))
    results['unforced_bubble_02_seconds']=d
    assert d['radial_relative_rms']<.005
    assert max(abs(np.array(d['enclosed_volume'])/d['initial_volume']-1))<1e-4
    internal=rng.normal(size=s.ne)*1e-4;s.internal.assign(internal);s.rhs_n.assign(internal)
    s.net_force.zero_();s.normal_tensor.zero_()
    s.launch(K.force_balance_reduce,s.ne,s.n,s.area,s.eg,s.internal,s.net_force,s.normal_tensor)
    s.launch(K.force_balance_apply,s.ne,s.n,s.area,s.eg,s.net_force,s.normal_tensor,1.,s.rhs_n)
    residual=np.linalg.norm((s.rhs_n.numpy()[:,None]*s.n.numpy()).sum(0))
    results['closed_surface_internal_force_balance']=float(residual)
    assert residual<1e-14
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(passed=True,checks=results),indent=2)+'\n')
    print(json.dumps(dict(passed=True,checks=results),indent=2))

if __name__=='__main__':main()
