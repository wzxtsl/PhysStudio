"""Continuous plane-contact geometry and independent distributed-energy checks."""
import argparse,json
from pathlib import Path
import numpy as np
import warp as wp
from . import kernels as K

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();wp.init();device='cuda:0'
    yz=np.array([(u,v) for u in np.linspace(-.01,.01,11) for v in np.linspace(-.01,.01,11)]);N=len(yz)
    pos=np.concatenate([np.column_stack([np.full(N,-.001),yz]),np.column_stack([np.full(N,.001),yz])])
    normals=np.concatenate([np.tile([1.,0,0],(N,1)),np.tile([-1.,0,0],(N,1))]);groups=np.repeat([0,1],N);size=2*N;cap=256;h=.012
    def arr(v,t=K.D):return wp.array(v,dtype=t,device=device)
    def zero(shape,t=K.D):return wp.zeros(shape,dtype=t,device=device)
    x=arr(pos,K.V);n=arr(normals,K.V);g=arr(groups,int);xf=arr(pos,wp.vec3);grid=wp.HashGrid(32,32,32,device=device);grid.build(xf,h)
    other=zero(size,int);direction=zero(size,K.V);gap=zero(size);share=zero(size);nb=zero((size,cap),int);count=zero(size,int);weights=zero((size,cap))
    wp.launch(K.continuous_contacts,size,inputs=[grid.id,x,n,g,h,cap,other,direction,gap,share,nb,count,weights],device=device)
    plane_error=float(np.max(np.abs(gap.numpy()+.002)));assert plane_error<1e-9 and np.all(other.numpy()>=0)
    area=arr(np.full(size,4e-6));diag=zero(size);rhs=zero(size);strength=zero(size);dt=.00025;stiffness=100000.;separation=.00015
    wp.launch(K.continuous_contact_rhs,size,inputs=[other,nb,count,weights,direction,gap,share,n,area,dt,stiffness,separation,1,diag,rhs,strength],device=device)
    neighbors=nb.numpy();counts=count.numpy();w=weights.numpy();d=direction.numpy();J=np.zeros((size,size))
    for i in range(size):
        J[i,i]=normals[i]@d[i]
        for k in range(counts[i]):J[i,neighbors[i,k]]-=w[i,k]*(normals[neighbors[i,k]]@d[i])
    c=strength.numpy();A=dt*dt*J.T@(c[:,None]*J);b=-dt*J.T@(c*(gap.numpy()+separation));rng=np.random.default_rng(32);v=rng.normal(size=size);out=zero(size)
    wp.launch(K.continuous_contact_matvec,size,inputs=[other,nb,count,weights,direction,n,strength,arr(v),out,dt,1.],device=device)
    operator_error=float(np.max(np.abs(out.numpy()-A@v)));force_error=float(np.max(np.abs(rhs.numpy()-b)));eig=float(np.linalg.eigvalsh(A)[0])
    assert operator_error<1e-18 and force_error<1e-18 and eig>-1e-18
    rhs.zero_();diag.zero_()
    wp.launch(K.continuous_contact_rhs,size,inputs=[other,nb,count,weights,direction,gap,share,n,area,dt,stiffness,separation,0,diag,rhs,strength],device=device)
    assert np.max(np.abs(rhs.numpy()))==0 and np.max(strength.numpy())==0
    gap.fill_(.0002)
    wp.launch(K.continuous_contact_rhs,size,inputs=[other,nb,count,weights,direction,gap,share,n,area,dt,stiffness,separation,0,diag,rhs,strength],device=device)
    unilateral_error=float(np.max(np.abs(rhs.numpy()+dt*J.T@(strength.numpy()*(gap.numpy()+separation)))))
    assert unilateral_error<1e-18 and np.min(strength.numpy())>0
    # Equal inner pressures cancel on an exactly shared planar interface.
    share.fill_(1.);capillary=zero(size,K.V);volume=arr([1.,1.]);gas=arr([103.,103.]);gamma=arr(np.full(size,(.07275-.03)/2478.95618937));test_rhs=arr(dt*3*area.numpy())
    wp.launch(K.multi_region_traction,size,inputs=[grid.id,x,n,area,gamma,capillary,other,share,nb,count,weights,1,g,gas,volume,.07275,2478.95618937,100.,h,dt,test_rhs],device=device)
    cancellation=float(np.max(np.abs(test_rhs.numpy())));assert cancellation<1e-16
    result=dict(passed=True,plane_projection_max_error_m=plane_error,contact_operator_max_error=operator_error,
                contact_force_max_error=force_error,smallest_eigenvalue=eig,equal_pressure_cancellation=cancellation,
                partition_of_unity_error=float(np.max(np.abs(w.sum(1)-1))),unilateral_force_max_error=unilateral_error,separated_unilateral_force_zero=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
