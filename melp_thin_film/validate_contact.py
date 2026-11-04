"""Check contact force and Hessian against the stated quadratic energy."""
import argparse,json
from pathlib import Path
import numpy as np
import warp as wp
from . import kernels as K

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();wp.init()
    rng=np.random.default_rng(12);N=10;other=np.arange(N)^1;n=rng.normal(size=(N,3));n/=np.linalg.norm(n,axis=1)[:,None]
    direction=n-n[other];direction/=np.linalg.norm(direction,axis=1)[:,None]
    gap=rng.uniform(-.001,.001,N);share=rng.uniform(.1,1,N);area=rng.uniform(1e-5,4e-5,N);dt=.0005;k=8000.;sep=.00015
    def arr(a,t=K.D):return wp.array(a,dtype=t,device='cuda:0')
    o=arr(other,int);normal=arr(n,K.V);directions=arr(direction,K.V);g=arr(gap);sh=arr(share);a=arr(area)
    diag=wp.zeros(N,dtype=K.D,device='cuda:0');rhs=wp.zeros_like(diag);strength=wp.zeros_like(diag)
    wp.launch(K.contact_rhs,N,inputs=[o,directions,g,sh,normal,a,dt,k,sep,diag,rhs,strength],device='cuda:0')
    J=np.zeros((N,N));c=k*.5*(area+area[other])*share
    for i,j in enumerate(other):J[i,i]=n[i]@direction[i];J[i,j]=-n[j]@direction[i]
    A=dt*dt*J.T@np.diag(c)@J;expected_rhs=-dt*J.T@(c*(gap+sep));x=rng.normal(size=N);z=wp.zeros_like(diag)
    wp.launch(K.contact_matvec,N,inputs=[o,directions,normal,strength,arr(x),z,dt,1.],device='cuda:0')
    operator_error=float(np.max(np.abs(z.numpy()-A@x)));force_error=float(np.max(np.abs(rhs.numpy()-expected_rhs)))
    eps=1e-4
    def energy(v):return .5*np.sum(c*(gap+sep+dt*J@v)**2)
    fd=np.array([(energy(eps*np.eye(N)[i])-energy(-eps*np.eye(N)[i]))/(2*eps) for i in range(N)])
    fd_error=float(np.max(np.abs(fd+expected_rhs)))
    result=dict(passed=True,operator_max_error=operator_error,force_max_error=force_error,finite_difference_gradient_max_error=fd_error,
                smallest_eigenvalue=float(np.linalg.eigvalsh(A)[0]),diagonal_max_error=float(np.max(np.abs(diag.numpy()-np.diag(A)))))
    assert operator_error<1e-18 and force_error<1e-18 and fd_error<1e-15 and result['smallest_eigenvalue']>-1e-18
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
