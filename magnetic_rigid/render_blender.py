"""Procedural Cycles studio render of recorded Warp states (no simulation here)."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
for env,sub in [('XDG_CACHE_HOME','blender'),('CUDA_CACHE_PATH','cuda'),('OPTIX_CACHE_PATH','optix')]:
    os.environ.setdefault(env,str(ROOT/'.cache'/sub))
import bpy
import numpy as np
from mathutils import Vector
S=10.


def material(name,color,rough=.3,metal=0.):
    m=bpy.data.materials.new(name);m.use_nodes=True
    p=m.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(*color,1)
    p.inputs['Roughness'].default_value=rough;p.inputs['Metallic'].default_value=metal
    return m


def finish(obj,name,mat,parent=None):
    obj.name=name
    if mat:obj.data.materials.append(mat)
    if parent:obj.parent=parent
    return obj


def box(name,p,size,mat,bevel=.008,parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1,location=np.array(p)*S)
    o=bpy.context.object;o.dimensions=np.array(size)*S
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    if bevel:
        mod=o.modifiers.new('Machined corners','BEVEL');mod.width=bevel*S;mod.segments=3
        o.modifiers.new('Corner normals','WEIGHTED_NORMAL')
    return finish(o,name,mat,parent)


def ball(name,p,r,mat,parent=None,segments=32,rings=20):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments,ring_count=rings,radius=r*S,location=np.array(p)*S)
    o=finish(bpy.context.object,name,mat,parent)
    for f in o.data.polygons:f.use_smooth=True
    return o


def cylinder(name,p,r,depth,mat,parent=None):
    bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=r*S,depth=depth*S,location=np.array(p)*S)
    o=finish(bpy.context.object,name,mat,parent)
    b=o.modifiers.new('Rim finish','BEVEL');b.width=min(.002,depth*.2)*S;b.segments=3
    o.modifiers.new('Corner normals','WEIGHTED_NORMAL')
    return o


def ring(name,p,r,tube,mat,parent=None,rotation=None):
    bpy.ops.mesh.primitive_torus_add(major_segments=64,minor_segments=10,major_radius=r*S,
                                   minor_radius=tube*S,location=np.array(p)*S)
    o=finish(bpy.context.object,name,mat,parent)
    if rotation:o.rotation_euler=rotation
    for f in o.data.polygons:f.use_smooth=True
    return o


def curve(name,points,radius,mat,parent=None):
    data=bpy.data.curves.new(name,'CURVE');data.dimensions='3D';data.resolution_u=1
    data.bevel_depth=radius*S;data.bevel_resolution=3
    sp=data.splines.new('POLY');sp.points.add(len(points)-1)
    for p,co in zip(sp.points,points):p.co=(*[v*S for v in co],1.)
    obj=bpy.data.objects.new(name,data);bpy.context.collection.objects.link(obj)
    return finish(obj,name,mat,parent)


def label(text,p,size,mat,rotation=(0,0,0),align='CENTER'):
    c=bpy.data.curves.new(text,'FONT');c.body=text;c.size=size*S;c.align_x=align;c.extrude=.0001*S
    o=bpy.data.objects.new(text,c);bpy.context.collection.objects.link(o);o.location=np.array(p)*S
    o.rotation_euler=rotation;o.data.materials.append(mat);return o


def area(name,p,target,power,size,size_y,color=(1,1,1)):
    d=bpy.data.lights.new(name,'AREA');d.energy=power;d.shape='RECTANGLE';d.size=size*S;d.size_y=size_y*S;d.color=color
    o=bpy.data.objects.new(name,d);bpy.context.collection.objects.link(o);o.location=np.array(p)*S
    o.rotation_euler=(Vector(np.array(target)*S)-o.location).to_track_quat('-Z','Y').to_euler()


def needle(parent,m):
    # A jeweled pointer fitted inside the sphere-union proxy's outer silhouette.
    for sign,key in [(1,'red'),(-1,'teal')]:
        verts=np.array([[0,-.012,0],[0,.012,0],[sign*.046,0,0],[0,0,.009],[0,0,-.004]])*S
        faces=[(0,2,3),(2,1,3),(0,4,2),(2,4,1),(0,3,1),(0,1,4)]
        mesh=bpy.data.meshes.new('Needle half');mesh.from_pydata(verts.tolist(),[],faces);mesh.update()
        o=bpy.data.objects.new('North' if sign==1 else 'South',mesh);bpy.context.collection.objects.link(o)
        finish(o,o.name,m[key],parent)
    ball('Pivot jewel',[0,0,.009],.0055,m['gold'],parent)


def coil(parent,r,m,led):
    ball('Soft iron pole',[0,0,0],r,m['dark'],parent)
    for z in np.linspace(-r*.48,r*.48,13):
        rr=math.sqrt(r*r-z*z)-.0015
        ring('Copper winding',[0,0,z],rr,.0017,m['copper'],parent)
    ring('Energized indicator',[0,0,r*.73],r*.68,.0025,led,parent)
    cylinder('Pole cap',[0,0,r*.94],r*.28,.004,m['steel'],parent)


def setup(cfg,a):
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.samples=a.samples
    scene.render.use_persistent_data=True
    scene.cycles.use_denoising=True;scene.cycles.denoiser='OPENIMAGEDENOISE';scene.cycles.denoising_use_gpu=True
    scene.cycles.max_bounces=16;scene.cycles.transmission_bounces=12;scene.cycles.glossy_bounces=8
    scene.cycles.transparent_max_bounces=12;scene.cycles.adaptive_threshold=.025;scene.cycles.seed=29
    pref=bpy.context.preferences.addons['cycles'].preferences;pref.compute_device_type='OPTIX';pref.get_devices()
    for d in pref.devices:d.use=d.type=='OPTIX'
    scene.cycles.device='GPU'
    scene.render.resolution_x=a.width;scene.render.resolution_y=a.height;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
    scene.view_settings.view_transform='AgX';scene.view_settings.look='AgX - Medium High Contrast';scene.view_settings.exposure=.35
    world=bpy.data.worlds.new('Soft studio');world.use_nodes=True;scene.world=world
    world.node_tree.nodes['Background'].inputs['Color'].default_value=(.26,.31,.36,1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value=.40
    m={k:material(k,c,r,metal) for k,c,r,metal in [
        ('steel',(.58,.67,.72),.19,.90),('brass',(.52,.31,.10),.23,.85),
        ('cobalt',(.025,.16,.28),.24,.70),('gold',(.64,.39,.12),.25,.8),
        ('copper',(.61,.22,.075),.23,.85),('dark',(.025,.047,.061),.26,.6),
        ('navy',(.018,.040,.059),.32,.25),('porcelain',(.63,.69,.70),.27,.08),
        ('ivory',(.82,.85,.80),.22,.25),('red',(.62,.035,.018),.23,.40),
        ('teal',(.015,.25,.28),.23,.45),('floor',(.34,.38,.40),.65,0) ]}
    led=material('Coil current',(.015,.35,.23),.22,.2)
    p=led.node_tree.nodes['Principled BSDF'];p.inputs['Emission Color'].default_value=(.02,.75,.48,1)
    roots=[];cables=[]
    for i,b in enumerate(cfg['bodies']):
        o=bpy.data.objects.new(b['name'],None);bpy.context.collection.objects.link(o);o.rotation_mode='QUATERNION';roots.append(o)
        if i==0:continue
        if b['name']=='Compass':needle(o,m)
        elif b.get('color')=='coil':coil(o,b['shape'][0]['r'],m,led)
        elif b.get('color')=='magnet':
            box('Permanent magnet',[0,0,0],[.28,.067,.067],m['ivory'],.014,o)
            box('North pole',[.092,0,0],[.105,.069,.069],m['red'],.014,o)
            box('South pole',[-.092,0,0],[.105,.069,.069],m['teal'],.014,o)
            ring('Rotor bearing',[0,0,-.042],.041,.006,m['gold'],o)
            for x,text in [(.10,'N'),(-.10,'S')]:
                mark=label(text,[x,-.009,.036],.022,m['ivory']);mark.parent=o
        else:
            for s in b.get('shape',[]):ball(b['name'],s['p'],s['r'],m.get(b.get('color'),m['steel']),o)
    name=cfg['scene']
    if name=='compass_garden':
        box('Instrument block',[0,0,-.04],[1.83,1.57,.12],m['navy'],.035)
        box('Porcelain face',[0,0,.023],[1.76,1.50,.016],m['porcelain'],.025)
        for b in cfg['bodies'][2:]:
            x,y,_=b['pos'];cylinder('Compass seat',[x,y,.040],.063,.012,m['ivory'])
            ring('Engraved bezel',[x,y,.047],.059,.0015,m['gold'])
            cylinder('Pivot pin',[x,y,.060],.004,.028,m['gold'])
            for k in range(4):
                ang=k*math.pi/2
                tick=box('Cardinal tick',[x+.053*math.cos(ang),y+.053*math.sin(ang),.047],[.008,.0015,.001],m['navy'],0)
                tick.rotation_euler[2]=ang
        cylinder('Magnet bearing',[0,0,.046],.062,.03,m['dark'])
        label('F I E L D   O F   N E E D L E S',[0,-.713,.032],.023,m['navy'])
        label('01   /   MAGNETIC RIGID BODIES',[0,.704,.032],.016,m['navy'])
        floor=-.12;target=[0,0,0]
    elif name=='magnetic_crane':
        box('Foundry base',[0,0,-.087],[2.42,1.70,.15],m['navy'],.04)
        box('Contact floor',[0,0,-.006],[2.36,1.64,.012],m['porcelain'],.003)
        for sign in [-1,1]:
            box('Side guard',[sign*1.039,0,.04],[.018,1.436,.08],m['dark'],.002)
            box('Front guard',[0,sign*.709,.04],[2.06,.018,.08],m['dark'],.002)
            box('Guard inlay',[0,sign*.709,.082],[2.06,.007,.003],m['gold'],.001)
        for x in [-.40,.43]:
            ring('Loading zone',[x,0,.0008],.275,.0012,m['gold'])
            for y in [-.32,.32]:box('Zone marker',[x,y,.001],[.48,.002,.001],m['gold'],0)
        for x in [-1.13,1.13]:
            box('Gantry footing',[x,.50,.03],[.12,.26,.06],m['dark'])
            box('Gantry column',[x,.50,.73],[.065,.08,1.4],m['dark'])
            box('Column inlay',[x,.455,.73],[.017,.004,1.27],m['gold'],.001)
        box('Bridge',[0,.50,1.42],[2.35,.11,.09],m['dark'])
        box('Forward rail',[0,0,1.42],[2.30,.055,.055],m['steel'])
        for x in [-1.13,1.13]:box('Rail bracket',[x,.25,1.42],[.065,.55,.065],m['dark'])
        cylinder('Moving cable guide',[0,0,0],.026,.026,m['gold'],roots[1])
        cables=[curve('Load cable',[[0,0,0],[0,0,1]],.0028,m['steel']),
                curve('Hoist cable',[[0,0,0],[0,0,1]],.002,m['steel'])]
        label('E L E C T R O M A G N E T I C   F O U N D R Y',[0,-.471,.002],.023,m['navy'])
        label('PICKUP',[-.40,-.36,.002],.022,m['navy']);label('RELEASE',[.43,-.36,.002],.022,m['navy'])
        floor=-.19;target=[0,0,.58]
    else:
        box('Cabinet plinth',[.08,0,-.225],[1.48,1.04,.10],m['navy'],.04)
        for y in [-.13,.13]:
            for x in [-.052,.052]:
                cylinder('Glass cradle leg',[x,y,-.048],.009,.258,m['dark'])
        ring('Glass cradle',[0,0,.086],.140,.006,m['gold'])
        glass=material('Thin optical glass',(.96,.985,1.),.035)
        p=glass.node_tree.nodes['Principled BSDF'];p.inputs['Transmission Weight'].default_value=1.;p.inputs['IOR'].default_value=1.46
        outer=ball('Optical shell',[0,0,.47],.409,glass,segments=128,rings=96)
        # Actual inner radius .405; solidify creates a closed thin glass shell.
        mod=outer.modifiers.new('Glass wall','SOLIDIFY');mod.thickness=.004*S;mod.offset=-1
        for y in [-.14,.14]:
            points=[[.59*math.sin(t),y,.47-.59*math.cos(t)] for t in np.linspace(0,math.pi,90)]
            curve('Electromagnet guide',points,.004,m['gold'])
        label('M A G N E T I C   C A B I N E T',[.08,-.445,-.173],.025,m['ivory'])
        floor=-.30;target=[.035,0,.40]
    box('Studio ground',[0,0,floor-.06],[200,200,.1],m['floor'],0)
    area('Key',[-1.8,-2.0,3.0],target,5400,2.0,2.2)
    area('Cool rim',[1.2,1.7,2.5],target,5250,1.0,1.8,(.76,.88,1.))
    area('Warm fill',[-1.7,.9,1.8],target,3000,1.0,1.8,(1.,.86,.68))
    area('Long reflection',[.2,-2.0,1.4],target,1500,1.7,.18)
    bpy.ops.object.camera_add();camera=bpy.context.object;scene.camera=camera;camera.data.lens=42
    camera.data.clip_start=.01;camera.data.clip_end=3000
    return scene,roots,cables,led,camera


def update(cfg,state,roots,cables,led,camera):
    for o,x,q in zip(roots,state['x'],state['q']):
        o.location=x*S;o.rotation_quaternion=(q[3],q[0],q[1],q[2])
    if cables:
        anchor=state['x'][1];tip=roots[2].matrix_world @ Vector((0,0,.10*S))
        # Evaluate the transform explicitly; depsgraph updates are deferred until render.
        from mathutils import Quaternion
        q=state['q'][2];tip=np.array(Quaternion((q[3],*q[:3]))@Vector((0,0,.10)))+state['x'][2]
        for cable,pair in zip(cables,[[anchor,tip],[[anchor[0],0,1.42],anchor]]):
            for p,co in zip(cable.data.splines[0].points,pair):p.co=(*[float(v*S) for v in co],1)
    strength=float(np.linalg.norm(state['permanent'],axis=1).max(initial=0))
    led.node_tree.nodes['Principled BSDF'].inputs['Emission Strength'].default_value=min(strength/1400,1)*3
    t=float(state['time']);f=t/cfg['duration'];name=cfg['scene']
    if name=='compass_garden':
        camera.data.lens=42
        angle=math.radians(-73+37*f);r=3.05;loc=[r*math.cos(angle),r*math.sin(angle),2.85];target=[0,0,.02]
    elif name=='magnetic_crane':
        camera.data.lens=32
        angle=math.radians(-78+28*f);r=3.70;loc=[r*math.cos(angle),r*math.sin(angle),1.95];target=[0,0,.55]
    else:
        camera.data.lens=38
        angle=math.radians(-73+22*f);r=3.30;loc=[r*math.cos(angle),r*math.sin(angle),1.52];target=[.05,0,.37]
    camera.location=np.array(loc)*S
    camera.rotation_euler=(Vector(np.array(target)*S)-camera.location).to_track_quat('-Z','Y').to_euler()


def main():
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    p=argparse.ArgumentParser();p.add_argument('--simulation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--indices');p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int)
    p.add_argument('--stride',type=int,default=1);p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=720)
    p.add_argument('--samples',type=int,default=32);p.add_argument('--resume',action='store_true')
    a=p.parse_args(argv);cfg=json.loads((a.simulation/'config.json').read_text());a.output.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).read_bytes();(a.output/f'renderer_snapshot_{a.start:05d}.py').write_bytes(source)
    files=sorted((a.simulation/'states').glob('*.npz'));indices=[int(v) for v in a.indices.split(',')] if a.indices else range(a.start,min(a.end or len(files),len(files)),a.stride)
    (a.output/f'render_settings_{a.start:05d}.json').write_text(json.dumps(dict(simulation=str(a.simulation.resolve()),width=a.width,height=a.height,
        samples=a.samples,indices=list(indices),renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    scene,roots,cables,led,camera=setup(cfg,a)
    for i in indices:
        target=a.output/f'{i:05d}.png'
        if a.resume and target.exists():continue
        state=np.load(a.simulation/'states'/f'{i:05d}.npz');update(cfg,state,roots,cables,led,camera)
        scene.render.filepath=str(target.resolve());start=time.monotonic();bpy.ops.render.render(write_still=True)
        print(json.dumps(dict(frame=i,seconds=time.monotonic()-start,path=str(target))),flush=True)

if __name__=='__main__':main()
