"""Studio views of ice growing from cooled boundaries."""
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'.cache/blender'))
os.environ.setdefault('OPTIX_CACHE_PATH',str(ROOT/'.cache/optix'))
import bpy
import numpy as np
from mathutils import Vector
from ice_melting import render_blender as art
from implicit_mixture.render_blender import lathe
from ice_solidification.geometry import star_vertices
from ice_solidification.phase_material import PhaseMaterial


def setup(cfg,args):
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.samples=args.samples
    scene.cycles.use_denoising=True;scene.cycles.denoiser='OPENIMAGEDENOISE';scene.cycles.denoising_use_gpu=True
    scene.cycles.max_bounces=14;scene.cycles.transmission_bounces=12;scene.cycles.glossy_bounces=6
    scene.cycles.transparent_max_bounces=12;scene.cycles.volume_bounces=3;scene.cycles.adaptive_threshold=.015
    scene.cycles.seed=19;scene.cycles.use_animated_seed=False
    pref=bpy.context.preferences.addons['cycles'].preferences;pref.compute_device_type='OPTIX';pref.get_devices()
    for device in pref.devices:device.use=device.type=='OPTIX'
    scene.cycles.device='GPU'
    scene.render.resolution_x=args.width;scene.render.resolution_y=args.height;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
    scene.view_settings.view_transform='AgX';scene.view_settings.look='AgX - Medium High Contrast';scene.view_settings.exposure=.25
    world=bpy.data.worlds.new('Neutral studio');scene.world=world;world.use_nodes=True
    world.node_tree.nodes['Background'].inputs['Color'].default_value=(.19,.20,.22,1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value=.35
    scale=cfg.get('render_scale',45.);art.SCALE=scale
    steel=art.material('Brushed steel',(.32,.36,.39),.24,.85)
    graphite=art.material('Graphite',(.035,.045,.055),.32,.3)
    copper=art.material('Cold copper',(.39,.20,.10),.28,.8)
    porcelain=art.material('Ceramic',(.24,.28,.30),.30)
    glass=art.transparent('Borosilicate',1.46,(.96,.985,.99),.025,.04,0.)
    water=art.transparent('Water',1.333,(.85,.96,.99),.025,.13,0.)
    ice=art.transparent('Ice',1.309,(.72,.86,.95),.14,.22,7.0)
    # World-space optical detail stays fixed as the freezing surface grows.
    position=ice.node_tree.nodes.new('ShaderNodeNewGeometry')
    for node in ice.node_tree.nodes:
        if node.bl_idname=='ShaderNodeTexNoise':
            ice.node_tree.links.new(position.outputs['Position'],node.inputs['Vector'])
        elif node.bl_idname=='ShaderNodeBump':
            node.inputs['Strength'].default_value=.045;node.inputs['Distance'].default_value=.006
        elif node.bl_idname=='ShaderNodeValToRGB':
            node.color_ramp.elements[0].color=(.20,.20,.20,1)
    art.cube('Studio floor',(0,0,-.24),(200,200,.12),art.material('Neutral floor',(.14,.16,.18),.65),0.)
    if cfg['scene']=='ice_cast':
        radius=cfg['star_radius']*scale;inner=star_vertices(radius);outer=star_vertices(radius+.10)
        height=.022*scale;vertices=[]
        for z,polygon in [(0.,inner),(height,inner),(0.,outer),(height,outer)]:
            vertices.extend([(float(p[0]),float(p[1]),z) for p in polygon])
        faces=[]
        for i in range(10):
            j=(i+1)%10
            faces.extend([(i,j,10+j,10+i),(20+j,20+i,30+i,30+j),
                          (10+i,10+j,30+j,30+i),(j,i,20+i,20+j)])
        faces=[tuple(reversed(face)) for face in faces]
        triangles=np.array([(q[0],q[1],q[2]) for q in faces]+[(q[0],q[2],q[3]) for q in faces])
        points=np.asarray(vertices)[triangles]
        signed_volume=np.einsum('ij,ij->i',points[:,0],np.cross(points[:,1],points[:,2])).sum()/6
        if signed_volume<=0:raise ValueError('Mold normals must face out of the glass volume')
        mesh=bpy.data.meshes.new('Star mold');mesh.from_pydata(vertices,[],faces);mesh.update()
        mold=bpy.data.objects.new('Transparent star mold',mesh);bpy.context.collection.objects.link(mold)
        mold.data.materials.append(glass)
        bevel=mold.modifiers.new('Polished mold edges','BEVEL');bevel.width=.012;bevel.segments=3
        mold.modifiers.new('Corner normals','WEIGHTED_NORMAL')
        art.cylinder('Cold casting plate',(0,0,-.055),radius+.18,.11,steel)
        art.cylinder('Insulated casting base',(0,0,-.15),radius+.23,.08,graphite)
    elif cfg['scene']=='cold_finger':
        r=cfg['bowl_radius']*scale;top=.055*scale;thick=.0018*scale
        lathe('Glass bowl',[(0,-.05),(r,-.05),(r+thick,0),(r+thick,top),(r,top),(r,0),(0,0)],glass)
        art.cylinder('Graphite pedestal',(0,0,-.13),r+.15,.12,graphite)
        center=np.array(cfg['rod_center'])*scale;rr=cfg['rod_radius']*scale
        art.cylinder('Copper cold finger',(center[0],center[1],.036*scale),rr,.072*scale,copper)
        art.cylinder('Insulated collar',(center[0],center[1],.077*scale),rr*1.55,.010*scale,graphite)
        for a in np.linspace(0,2*math.pi,6,endpoint=False):
            art.cylinder('Rim bolt',((r+.11)*math.cos(a),(r+.11)*math.sin(a),-.065),.035,.035,steel)
    elif cfg['scene']=='cascade':
        art.cube('Cold base',(0,0,-.07),(.184*scale,.068*scale,.14),steel,.04)
        art.cube('Upper cooling step',(-.0575*scale,0,.013*scale),(.065*scale,.064*scale,.026*scale),steel,.025)
        art.cube('Middle cooling step',(0,0,.007*scale),(.05*scale,.064*scale,.014*scale),steel,.025)
        for sign in [-1,1]:
            art.cube('Glass side',(0,sign*.033*scale,.037*scale),(.184*scale,.002*scale,.074*scale),glass,.012)
        art.cube('Back glass',(-.091*scale,0,.045*scale),(.002*scale,.068*scale,.09*scale),glass,.012)
        art.cube('Front glass',(.091*scale,0,.020*scale),(.002*scale,.068*scale,.04*scale),glass,.012)
    else:
        radius=cfg.get('plate_radius',.019)
        art.cylinder('Freezing plate',(0,0,-.05),radius*scale,.10,steel)
        art.cylinder('Insulated base',(0,0,-.14),radius*scale+.15,.10,graphite)
        for r in np.linspace(radius*.65,radius*.95,5):
            bpy.ops.mesh.primitive_torus_add(major_radius=r*scale,minor_radius=.004,major_segments=128,minor_segments=8,location=(0,0,.001))
            bpy.context.object.data.materials.append(steel)
    art.area('Large softbox',(-6,-5,11),(0,0,1),1800,6.,5.)
    art.area('Ice rim',(5,4,8),(0,0,1),2200,3.,6.,(.91,.96,1.))
    art.area('Fill',(-6,3,6),(0,0,1),1300,5.,5.)
    art.area('Reflection strip',(1,-6,5),(0,0,1),700,4.,.6)
    iceobj=art.dynamic('Growing ice',ice);waterobj=art.dynamic('Liquid water',water)
    iceobj.modifiers.clear();waterobj.modifiers.clear()
    bpy.ops.object.camera_add();camera=bpy.context.object;scene.camera=camera;camera.data.lens=55
    camera.data.clip_start=.01;camera.data.clip_end=300
    return scene,iceobj,waterobj,camera,scale


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--meshes',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,nargs='*');p.add_argument('--width',type=int,default=1280)
    p.add_argument('--height',type=int,default=800);p.add_argument('--samples',type=int,default=48)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:]);cfg=json.loads((args.input/'config.json').read_text())
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'frames').mkdir()
    frames=args.frames or list(range(cfg['frames']))
    settings=dict(input=str(args.input.resolve()),meshes=str(args.meshes.resolve()),frames=frames,
        width=args.width,height=args.height,samples=args.samples,source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'render_config.json').write_text(json.dumps(settings,indent=2))
    scene,ice,water,camera,scale=setup(cfg,args);started=time.monotonic()
    phase_material=None
    if cfg.get('phase_rendering')=='continuous':
        phase_material=PhaseMaterial(np.load(args.meshes/f'{frames[0]:05d}.npz'),scale)
        water.data.materials.clear();water.data.materials.append(phase_material.material)
        water.name='Continuous ice and liquid';ice.hide_render=True
    for index in frames:
        data=np.load(args.meshes/f'{index:05d}.npz')
        if phase_material:
            phase_material.update(data)
            art.set_mesh(water,data['fluid_vertices'],data['fluid_faces'])
        else:
            art.set_mesh(ice,data['ice_vertices'],data['ice_faces']);art.set_mesh(water,data['water_vertices'],data['water_faces'])
        t=index/(cfg['frames']-1);angle=math.radians(cfg['camera_start']+cfg['camera_orbit']*t)
        radius=15.5 if cfg['scene']=='cascade' else 12.
        elevation=9.0 if cfg['scene']=='cascade' else (8.3 if cfg['scene']=='cold_finger' else 7.)
        if cfg['scene']=='ice_cast':radius=9.5;elevation=9.
        camera.location=(radius*math.cos(angle),radius*math.sin(angle),elevation+.4*math.sin(math.pi*t))
        target=Vector(np.array(cfg['target'])*scale)
        camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler()
        scene.render.filepath=str((args.output/'frames'/f'{index:05d}.png').resolve());bpy.ops.render.render(write_still=True)
        print(json.dumps(dict(frame=index,elapsed=time.monotonic()-started)),flush=True)


if __name__=='__main__':main()
