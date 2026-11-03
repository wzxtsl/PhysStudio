"""Render mature tree trajectories with branch-bound, independently turning leaves."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'.cache/blender'))
os.environ.setdefault('OPTIX_CACHE_PATH',str(ROOT/'.cache/optix'))
import bpy
import numpy as np
from mathutils import Vector
from rhizomorph import render_blender as botany
from ice_melting import render_blender as art
from windy_trees.flow import Pathlines,WindStreaks

def flow_object():
    material=bpy.data.materials.new('Air speed pathlines');material.use_nodes=True
    nt=material.node_tree;nt.nodes.clear()
    attribute=nt.nodes.new('ShaderNodeAttribute');attribute.attribute_name='wind_color'
    emission=nt.nodes.new('ShaderNodeEmission');emission.inputs['Strength'].default_value=.65
    transparent=nt.nodes.new('ShaderNodeBsdfTransparent');mix=nt.nodes.new('ShaderNodeMixShader')
    output=nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(attribute.outputs['Color'],emission.inputs['Color'])
    nt.links.new(attribute.outputs['Alpha'],mix.inputs[0])
    nt.links.new(transparent.outputs[0],mix.inputs[1]);nt.links.new(emission.outputs[0],mix.inputs[2])
    nt.links.new(mix.outputs[0],output.inputs['Surface'])
    obj=botany.mesh_object('SPH wind pathlines',[],[],material)
    obj.visible_shadow=False
    return obj

def update_flow(obj,pathlines,frame):
    vertices,faces,colors=pathlines.mesh(frame)
    botany.replace_mesh(obj,vertices,faces)
    if len(colors):
        attribute=obj.data.color_attributes.new(name='wind_color',type='FLOAT_COLOR',domain='POINT')
        attribute.data.foreach_set('color',colors.ravel())

def inlet_guide():
    """A visual cross-section at the downstream edge of the inlet relaxation zone."""
    x=-4.72;y=2.65;bottom=.20;top=5.25
    frame=art.material('Inlet frame',(.055,.069,.073),.5)
    for yy in [-y,y]:art.cube('Inlet upright',(x,yy,(top+bottom)/2),(.045,.045,top-bottom),frame,.012)
    for zz in [bottom,top]:art.cube('Inlet rail',(x,0,zz),(.045,2*y,.045),frame,.012)
    pane=bpy.data.materials.new('Inlet guide surface');pane.use_nodes=True;nt=pane.node_tree;nt.nodes.clear()
    trans=nt.nodes.new('ShaderNodeBsdfTransparent');emission=nt.nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value=(.10,.125,.13,1);emission.inputs['Strength'].default_value=.3
    mix=nt.nodes.new('ShaderNodeMixShader');mix.inputs[0].default_value=.28
    output=nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(trans.outputs[0],mix.inputs[1]);nt.links.new(emission.outputs[0],mix.inputs[2])
    nt.links.new(mix.outputs[0],output.inputs['Surface'])
    obj=botany.mesh_object('Wind inlet plane',[(x,-y,bottom),(x,y,bottom),(x,y,top),(x,-y,top)],[(0,1,2,3)],pane)
    obj.visible_shadow=False

def rotate(q,v):
    xyz=q[...,:3];w=q[...,3:4]
    t=2*np.cross(xyz,v)
    return v+w*t+np.cross(xyz,t)

def multiply(a,b):
    xyz=a[...,:3]*b[...,3:4]+b[...,:3]*a[...,3:4]+np.cross(a[...,:3],b[...,:3])
    w=a[...,3:4]*b[...,3:4]-np.sum(a[...,:3]*b[...,:3],axis=-1,keepdims=True)
    return np.concatenate([xyz,w],axis=-1)

def assign_vertices(obj,vertices):
    obj.data.vertices.foreach_set('co',np.asarray(vertices,dtype=np.float32).ravel())
    obj.data.update()

def setup(growth,cfg,args,source):
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    sc=bpy.context.scene;sc.render.engine='CYCLES';sc.cycles.samples=args.samples
    sc.render.use_persistent_data=True
    sc.cycles.use_denoising=True;sc.cycles.denoiser='OPENIMAGEDENOISE';sc.cycles.denoising_use_gpu=True
    pref=bpy.context.preferences.addons['cycles'].preferences;pref.compute_device_type='OPTIX';pref.get_devices()
    for dev in pref.devices:dev.use=dev.type=='OPTIX'
    sc.cycles.device='GPU';sc.cycles.seed=73;sc.cycles.use_animated_seed=False
    sc.cycles.max_bounces=8;sc.cycles.transparent_max_bounces=8
    sc.render.resolution_x=args.width;sc.render.resolution_y=args.height;sc.render.resolution_percentage=100
    sc.render.image_settings.file_format='PNG';sc.render.image_settings.color_mode='RGB'
    sc.view_settings.view_transform='AgX';sc.view_settings.look='AgX - Medium High Contrast'
    world=bpy.data.worlds.new('Cool studio');world.use_nodes=True;sc.world=world
    world.node_tree.nodes['Background'].inputs['Color'].default_value=(.32,.39,.44,1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value=.28
    floor=art.material('Slate',(0.032,.044,.049),.7)
    art.cube('Studio ground',(0,0,-.5),(200,200,.2),floor,0)
    soil=botany.wood_material('Earth',[(.016,.012,.008),(.065,.048,.028)],root=True)
    for node in soil.node_tree.nodes:
        if node.bl_idname=='ShaderNodeVectorMath':node.inputs[1].default_value=(16,16,16)
    bpy.ops.mesh.primitive_cylinder_add(vertices=128,radius=3.05,depth=.34,location=(0,0,-.21))
    bed=bpy.context.object;bed.name='Rooted soil island';bed.data.materials.append(soil)
    bevel=bed.modifiers.new('Soil edge','BEVEL');bevel.width=.06;bevel.segments=3
    bed.modifiers.new('Weighted normals','WEIGHTED_NORMAL')
    bark=botany.wood_material('Bark',[(.09,.045,.018),(.28,.17,.085)])
    if cfg['scene']=='birch_lateral':
        bark=botany.wood_material('Birch bark',[(.065,.052,.035),(.75,.73,.65)],birch=True)
    leaf=botany.leaf_material(growth)
    bsdf=leaf.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Specular IOR Level'].default_value=.25;bsdf.inputs['Coat Weight'].default_value=.08
    twig=botany.mesh_object('Elastic crown',*botany.branches(source,0),bark)
    blade=botany.mesh_object('Living canopy',*botany.leaves(source,growth,growth['steps']),leaf)
    rootmat=botany.wood_material('Root bark',[(.16,.068,.025),(.46,.26,.11)],root=True)
    botany.mesh_object('Anchored roots',*botany.branches(source,1),rootmat)
    stone=botany.wood_material('Fieldstone',[(.055,.065,.065),(.18,.19,.17)],root=True)
    rng=np.random.default_rng(491)
    for i in range(40):
        angle=rng.uniform(0,2*np.pi);r=rng.uniform(.8,2.8)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2,radius=rng.uniform(.025,.10),location=(r*np.cos(angle),r*np.sin(angle),-.025))
        obj=bpy.context.object;obj.scale=(1.2,.8,.45);obj.data.materials.append(stone)
        for face in obj.data.polygons:face.use_smooth=True
    if args.flow=='streaks':inlet_guide()
    art.area('Daylight',(-4,-6,8),(0,0,2),1700,6,6,(1,.98,.95))
    art.area('Canopy rim',(3,4,7),(0,0,2.5),1500,5,5,(.86,.94,1))
    art.area('Soft fill',(5,-2,4),(0,0,1.7),750,5,5,(.94,.97,1))
    bpy.ops.object.camera_add();cam=bpy.context.object;sc.camera=cam
    cam.data.type='ORTHO';cam.data.clip_end=250
    return sc,twig,blade,cam

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--width',type=int,default=1280)
    p.add_argument('--height',type=int,default=720);p.add_argument('--samples',type=int,default=24)
    p.add_argument('--indices',type=int,nargs='*');p.add_argument('--camera',choices=['orbit','static'],default='orbit')
    p.add_argument('--flow',choices=['streaks','pathlines','off'],default='streaks')
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:]);args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'frames').mkdir()
    cfg=json.loads((args.input/'config.json').read_text());growth=json.loads((args.input/'growth_config.json').read_text())
    with np.load(args.input/'asset.npz') as z:asset={k:z[k] for k in z.files}
    with np.load(args.input/'source.npz') as z:source={k:z[k] for k in z.files}
    sc,twigs,blades,cam=setup(growth,cfg,args,source)
    flow=flow_object() if args.flow!='off' else None
    pathlines=(WindStreaks if args.flow=='streaks' else Pathlines)(args.input/'states',cfg['fps'])
    verts,_=botany.leaves(source,growth,growth['steps'])
    rest=np.asarray(verts).reshape(-1,27,3)-source['p'][asset['leaf_ids'],None,:]
    shoot=asset['mapping']>=0
    states=sorted((args.input/'states').glob('*.npz'))
    top=float(source['p'][source['kind']==0,2].max());center=.5*(top-.15)
    vertical=top+1.65
    if args.flow=='streaks':
        top=max(top,5.25);center=.5*(top-.15)-.25;vertical=top+1.85
    indices=args.indices if args.indices is not None else range(len(states));start=time.monotonic()
    for frame in indices:
        with np.load(states[frame]) as z:state={k:z[k] for k in z.files}
        posed={**source,'p':source['p'].copy()};posed['p'][shoot]=state['p'][asset['mapping'][shoot]]
        vertices,_=botany.branches(posed,0);assign_vertices(twigs,vertices)
        half=state['leaf_theta']*.5
        local=np.concatenate([asset['leaf_across']*np.sin(half)[:,None],np.cos(half)[:,None]],axis=1)
        rotation=multiply(state['q'][asset['leaf_owner']],local)
        vertices=rotate(rotation[:,None,:],rest)+state['p'][asset['leaf_owner'],None,:]
        assign_vertices(blades,vertices.reshape(-1,3))
        if flow is not None:update_flow(flow,pathlines,frame)
        fraction=frame/max(1,len(states)-1)
        angle=math.radians(cfg['camera_start']+(cfg['camera_orbit']*fraction if args.camera=='orbit' else 0.))
        if args.flow=='streaks':
            angle=math.radians(cfg['camera_start']-20.+(.75*cfg['camera_orbit']*fraction if args.camera=='orbit' else 0.))
        cam.location=(15*math.cos(angle),15*math.sin(angle),center+3.4)
        target=Vector((-.45 if args.flow=='streaks' else .10,0,center))
        cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler()
        cam.data.ortho_scale=vertical*args.width/args.height
        sc.render.filepath=str((args.output/'frames'/f'{frame:05d}.png').resolve())
        bpy.ops.render.render(write_still=True)
        print(json.dumps(dict(frame=frame,elapsed=time.monotonic()-start)),flush=True)
    (args.output/'render.json').write_text(json.dumps(dict(frames=len(states),rendered=list(indices),
        width=args.width,height=args.height,samples=args.samples,camera=args.camera,flow=args.flow,
        persistent_data=sc.render.use_persistent_data,simulation=str(args.input.resolve()),
        flow_history_seconds=pathlines.history/cfg['fps'],recorded_air_particles=len(state['air_p'])),indent=2))

if __name__=='__main__':main()
