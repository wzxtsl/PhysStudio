"""Numerical mechanics, aerodynamic feedback and recorded-path validation."""
import argparse
import json
from pathlib import Path
import numpy as np
import warp as wp
from .tree import Tree, mass_action, accumulate_load, matrix_finish, global_rotations, forward
from .wind import Wind
from .flow import Pathlines
from .validate_flow import inspect as inspect_flow


def synthetic_asset(parent, position, stiffness=8., damping=.3):
    parent=np.asarray(parent,np.int32);position=np.asarray(position,np.float32)
    n=len(parent);edge=np.zeros_like(position);edge[1:]=position[1:]-position[parent[1:]]
    return dict(parent=parent,position=position,edge=edge,mass=np.ones(n,np.float32),
        radius=np.full(n,.03),length=np.linalg.norm(edge,axis=1),inertia=np.full(n,3.),
        stiffness=np.full(n,stiffness),damping=np.full(n,damping),
        leaf_owner=np.array([0],np.int32),leaf_direction=np.array([[1.,0,0]]),
        leaf_across=np.array([[0.,1,0]]),leaf_normal=np.array([[0.,0,1]]),
        leaf_size=np.array([.1]),leaf_area=np.array([0.]),leaf_inertia=np.array([.001]),
        leaf_stiffness=np.array([.01]),leaf_damping=np.array([.001]))


def finite_difference_jacobian(asset):
    """Differentiate finite rigid rotations of downstream nodes on the CPU."""
    parent=asset['parent'];p=asset['position'].astype(np.float64);n=len(p);epsilon=1.e-5
    jac=np.zeros((3*n,3*(n-1)))
    for joint in range(1,n):
        downstream=[]
        for i in range(n):
            j=i
            while j>=0 and j!=joint:j=parent[j]
            if j==joint:downstream.append(i)
        pivot=p[parent[joint]]
        for axis in range(3):
            unit=np.eye(3)[axis];offset=p[downstream]-pivot
            plus=offset*np.cos(epsilon)+np.cross(unit,offset)*np.sin(epsilon)
            plus+=np.outer(offset@unit,unit)*(1-np.cos(epsilon))
            minus=offset*np.cos(epsilon)-np.cross(unit,offset)*np.sin(epsilon)
            minus+=np.outer(offset@unit,unit)*(1-np.cos(epsilon))
            derivative=np.zeros_like(p);derivative[downstream]=(plus-minus)/(2*epsilon)
            jac[:,3*(joint-1)+axis]=derivative.ravel()
    return jac


def mechanics(device):
    asset=synthetic_asset([-1,0,1,2,2,1,5,0,7],
        [[0,0,0],[0,0,1],[.15,0,2],[.7,.1,2.8],[-.5,.25,2.5],
         [.3,.8,1.9],[.6,1.5,2.2],[-.6,-.3,.8],[-1.2,-.5,1.2]])
    tree=Tree(asset,device,iterations=60);n=tree.n;dt=1/240
    jac=finite_difference_jacobian(asset)
    mass=jac.T@(np.repeat(asset['mass'],3)[:,None]*jac)
    diagonal=np.repeat(dt*asset['damping'][1:]+dt*dt*asset['stiffness'][1:],3)
    matrix=mass+np.diag(diagonal)
    rng=np.random.default_rng(240);errors=[]
    for _ in range(5):
        vector=rng.normal(size=(n,3));vector[0]=0
        tree.search.assign(vector.astype(np.float32));tree.mf_total.zero_();tree.mm_total.zero_()
        wp.launch(mass_action,n,[tree.parent,tree.p,tree.q,tree.mass,tree.search,tree.mf,tree.mm],device=device)
        wp.launch(accumulate_load,n,[tree.parent,tree.mf,tree.mm,tree.mf_total,tree.mm_total],device=device)
        wp.launch(matrix_finish,n,[tree.parent,tree.p,tree.q,tree.mf_total,tree.mm_total,
            tree.stiffness,tree.damping,dt,tree.search,tree.ap],device=device)
        expected=matrix@vector[1:].ravel();actual=tree.ap.numpy()[1:].ravel()
        errors.append(float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)))
    old=rng.normal(size=(n,3))*.05;old[0]=0;force=rng.normal(size=(n,3))
    tree.clear();tree.omega.assign(old.astype(np.float32));tree.force.assign(force.astype(np.float32))
    tree.moment.assign(np.cross(asset['position'],force).astype(np.float32))
    expected=np.linalg.solve(matrix,mass@old[1:].ravel()+dt*jac.T@force.ravel())
    tree.advance(dt);actual=tree.omega.numpy()[1:].ravel()
    solve_error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    assert max(errors)<2.e-5,(errors,'Matrix-free mass action disagrees with finite rotations')
    assert solve_error<3.e-4,(solve_error,'PCG disagrees with a dense implicit step')
    return dict(matrix_action_relative_error=max(errors),implicit_step_relative_error=solve_error,
                minimum_system_eigenvalue=float(np.linalg.eigvalsh(matrix).min()))


def oscillator(device):
    asset=synthetic_asset([-1,0],[[0,0,0],[0,0,1]],stiffness=4.,damping=.4)
    tree=Tree(asset,device,iterations=4);theta=np.zeros((2,3),np.float32);theta[1,1]=.02
    tree.theta.assign(theta);dt=1/240
    wp.launch(global_rotations,tree.n,[tree.theta,tree.parent,tree.q],device=device)
    wp.launch(forward,tree.n,[tree.parent,tree.edge,tree.q,tree.p,tree.v,tree.base,dt],device=device)
    tree.v.zero_();angles=[.02];energy=[.5*4*.02**2]
    for _ in range(480):
        tree.clear();tree.advance(dt)
        angle=float(tree.theta.numpy()[1,1]);rate=float(tree.omega.numpy()[1,1])
        angles.append(angle);energy.append(.5*rate*rate+2*angle*angle)
    time=np.arange(len(angles))*dt;frequency=np.sqrt(4-.2**2)
    reference=.02*np.exp(-.2*time)*(np.cos(frequency*time)+.2/frequency*np.sin(frequency*time))
    error=float(np.sqrt(np.mean((np.asarray(angles)-reference)**2))/.02)
    rise=float(np.max(np.diff(energy)));assert error<.025,error;assert rise<1.e-8,rise
    return dict(normalized_rms_angle_error=error,max_step_energy_increase=rise,
                final_energy_ratio=float(energy[-1]/energy[0]),steps=480,dt=dt)


def pathlines(simulations):
    result={}
    for folder in simulations:
        cfg=json.loads((folder/'config.json').read_text());paths=Pathlines(folder/'states',cfg['fps'])
        frames=len(list((folder/'states').glob('*.npz')));max_error=0.;max_step=0.;count=0
        for frame in range(1,frames,17):
            positions,_=paths.state(frame)
            for curve in paths.curves(frame):
                max_error=max(max_error,float(np.linalg.norm(curve['points'][-1]-positions[curve['particle']])))
                max_step=max(max_step,float(np.linalg.norm(np.diff(curve['points'],axis=0),axis=1).max()))
                count+=1
        assert max_error<1.e-6 and max_step<1.,(folder,max_error,max_step)
        result[cfg['scene']]=dict(checked_curves=count,max_endpoint_error=max_error,
                                  max_interpolated_step=max_step,history_seconds=paths.history/cfg['fps'])
    return result


def wake(simulation,output,device):
    """Compare identical inflows with a fixed crown, with and without air feedback."""
    cfg=json.loads((simulation/'config.json').read_text());cfg.update(turn=0.,direction=0.,turbulence=0.)
    with np.load(simulation/'asset.npz') as data:asset={k:data[k] for k in data.files}
    dt=1/(cfg['fps']*cfg['substeps']);records={};fields={}
    xbins=np.linspace(-4,5,37);zbins=np.linspace(.1,5.5,28)
    for feedback in [False,True]:
        cfg['coupling']=feedback;tree=Tree(asset,device);air=Wind(cfg,device)
        velocity_sum=np.zeros((len(xbins)-1,len(zbins)-1));count=np.zeros_like(velocity_sum)
        measured=[]
        for step in range(round(6/dt)):
            t=step*dt;tree.clear();air.begin(t);air.couple(tree);air.advance(dt,t)
            if step*dt>=4. and step%8==0:
                p=air.x.numpy();v=air.v.numpy()
                mask=(abs(p[:,1])<1.25)&(p[:,0]>1.5)&(p[:,0]<3.5)&(p[:,2]>1)&(p[:,2]<4)
                if mask.any():measured.append(float(v[mask,0].mean()))
                slab=abs(p[:,1])<1.25
                velocity_sum+=np.histogram2d(p[slab,0],p[slab,2],bins=[xbins,zbins],weights=v[slab,0])[0]
                count+=np.histogram2d(p[slab,0],p[slab,2],bins=[xbins,zbins])[0]
        name='two_way' if feedback else 'one_way'
        records[name]=dict(mean_downstream_velocity=float(np.mean(measured)),samples=len(measured))
        fields[name]=np.divide(velocity_sum,count,out=np.full_like(count,np.nan),where=count>0)
        print(json.dumps(dict(wake=name,**records[name])),flush=True)
    np.savez_compressed(output/'wake_comparison.npz',**fields,x_edges=xbins,z_edges=zbins,
                         tree_p=asset['position'],tree_parent=asset['parent'])
    records['downstream_reduction_percent']=100*(1-records['two_way']['mean_downstream_velocity']/
                                                     records['one_way']['mean_downstream_velocity'])
    records['region_m']=dict(x=[1.5,3.5],y=[-1.25,1.25],z=[1.,4.])
    records['averaging_time_s']=[4.,6.];records['tree_motion']='fixed'
    assert records['downstream_reduction_percent']>0.,records
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--simulations',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);wp.init()
    result={}
    for name,func in [('mechanics',lambda:mechanics(args.device)),('oscillator',lambda:oscillator(args.device)),
                      ('pathlines',lambda:pathlines(args.simulations)),
                      ('filaments',lambda:{folder.name:inspect_flow(folder) for folder in args.simulations}),
                      ('wake',lambda:wake(args.simulations[0],args.output,args.device))]:
        result[name]=func();print(json.dumps({name:result[name]}),flush=True)
        (args.output/'validation.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':main()
