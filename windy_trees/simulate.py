"""Simulate a grown tree in two-way SPH wind and save mechanical trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import warp as wp
from .assets import prepare
from .scenes import scene_config,SCENES
from .tree import Tree
from .wind import Wind

def run(source,output,cfg,device='cuda:0'):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);(output/'states').mkdir()
    asset,original,growth,provenance=prepare(source,cfg)
    np.savez_compressed(output/'asset.npz',**asset);np.savez_compressed(output/'source.npz',**original)
    (output/'growth_config.json').write_text(json.dumps(growth,indent=2))
    (output/'config.json').write_text(json.dumps(cfg,indent=2))
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    (output/'source_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},indent=2))
    wp.init();tree=Tree(asset,device,iterations=cfg['solver_iterations']);wind=Wind(cfg,device);dt=1/(cfg['fps']*cfg['substeps'])
    start=time.monotonic();t=0.
    for _ in range(round(cfg['warmup']/dt)):
        wind.begin(t);wind.advance(dt,t);t+=dt
    frames=cfg.get('frames',round(cfg['seconds']*cfg['fps'])+1)
    audit=dict(frames=frames,mechanical_nodes=tree.n,leaf_blades=tree.nl,air_particles=wind.n,
               max_segment_length_error=0.,max_base_error=0.,max_coupling_force_residual=0.,max_displacement=0.,
               max_solver_residual=0.,max_air_speed=0.,max_joint_angle=0.)
    trace_count=int(cfg.get('trace_count',400))
    if trace_count<1:raise ValueError('trace_count must be positive')
    traces=np.arange(0,wind.n,max(1,wind.n//trace_count))
    np.save(output/'trace_particle_ids.npy',traces)
    audit['recorded_air_particles']=len(traces)
    with (output/'diagnostics.jsonl').open('w') as log:
        for frame in range(frames):
            if frame:
                for _ in range(cfg['substeps']):
                    tree.clear();wind.begin(t);wind.couple(tree);tree.advance(dt);wind.advance(dt,t);t+=dt
            state=tree.state();air_v=wind.v.numpy();air_p=wind.x.numpy()
            assert all(np.isfinite(value).all() for value in state.values()),('Invalid tree',frame)
            assert np.isfinite(air_v).all() and np.isfinite(air_p).all(),('Invalid air',frame)
            np.savez_compressed(output/'states'/f'{frame:05d}.npz',**state,
                air_p=air_p[traces],air_v=air_v[traces],time=frame/cfg['fps'])
            info=tree.diagnostics();tf=tree.force.numpy().sum(0,dtype=np.float64);af=wind.force.numpy().sum(0,dtype=np.float64)
            residual=float(np.linalg.norm(tf+af)/max(1.,np.linalg.norm(tf))) if cfg['coupling'] else 0.
            info.update(frame=frame,time=frame/cfg['fps'],tree_force=tf.tolist(),coupling_force_residual=residual,
                        mean_air_speed=float(np.linalg.norm(air_v,axis=1).mean()),max_air_speed=float(np.linalg.norm(air_v,axis=1).max()),
                        elapsed=time.monotonic()-start)
            for key,value in [('max_segment_length_error',info['max_segment_length_error']),('max_base_error',info['base_error']),
                              ('max_coupling_force_residual',residual),('max_displacement',info['max_displacement']),
                              ('max_solver_residual',info['relative_solver_residual']),('max_air_speed',info['max_air_speed']),
                              ('max_joint_angle',info['max_joint_angle'])]:
                audit[key]=max(audit[key],value)
            log.write(json.dumps(info)+'\n');log.flush()
            if frame%30==0:print(json.dumps(info),flush=True)
    audit['elapsed']=time.monotonic()-start
    assert audit['max_segment_length_error']<1.e-4 and audit['max_base_error']<1.e-6
    assert audit['max_coupling_force_residual']<1.e-4
    assert audit['max_air_speed']<100.,'Unresolved high-speed air motion'
    assert audit['max_joint_angle']<.8,'Joint rotations exceed the local tangent model range'
    (output/'audit.json').write_text(json.dumps(audit,indent=2))
    return audit

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',choices=SCENES,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--settings',type=Path);p.add_argument('--frames',type=int);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();cfg=scene_config(args.scene)
    if args.settings:cfg.update(json.loads(args.settings.read_text()))
    if args.frames is not None:cfg['frames']=args.frames
    run(args.source,args.output,cfg,args.device)

if __name__=='__main__':main()
