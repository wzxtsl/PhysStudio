"""Procedural 3D studio with angle/thickness-dependent thin-film reflection."""
import argparse,json,math,sys,os,time,hashlib
from pathlib import Path
import numpy as np
import bpy
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1]
for key,folder in [('XDG_CACHE_HOME','blender'),('CUDA_CACHE_PATH','cuda'),('OPTIX_CACHE_PATH','optix')]:os.environ.setdefault(key,str(ROOT/'.cache'/folder))
S=40.
RENDERER_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def material(name,color,metal=0.,rough=.3):
    m=bpy.data.materials.new(name);m.use_nodes=True;p=m.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value=(*color,1);p.inputs['Metallic'].default_value=metal;p.inputs['Roughness'].default_value=rough
    return m


def soap():
    m=bpy.data.materials.new('Air / aqueous film / air');m.use_nodes=True;nodes=m.node_tree.nodes;nodes.clear();links=m.node_tree.links
    def mathnode(op,a,b=None):
        n=nodes.new('ShaderNodeMath');n.operation=op
        for k,v in enumerate([a,b]):
            if v is None:continue
            if isinstance(v,(float,int)):n.inputs[k].default_value=v
            else:links.new(v,n.inputs[k])
        return n.outputs[0]
    geo=nodes.new('ShaderNodeNewGeometry');dot=nodes.new('ShaderNodeVectorMath');dot.operation='DOT_PRODUCT'
    links.new(geo.outputs['Normal'],dot.inputs[0]);links.new(geo.outputs['Incoming'],dot.inputs[1])
    cosine=mathnode('MAXIMUM',mathnode('ABSOLUTE',dot.outputs['Value']),.001)
    ct=mathnode('SQRT',mathnode('SUBTRACT',1.,mathnode('DIVIDE',mathnode('SUBTRACT',1.,mathnode('MULTIPLY',cosine,cosine)),1.333**2)))
    rs=mathnode('POWER',mathnode('DIVIDE',mathnode('SUBTRACT',cosine,mathnode('MULTIPLY',1.333,ct)),mathnode('ADD',cosine,mathnode('MULTIPLY',1.333,ct))),2.)
    rp=mathnode('POWER',mathnode('DIVIDE',mathnode('SUBTRACT',mathnode('MULTIPLY',1.333,cosine),ct),mathnode('ADD',mathnode('MULTIPLY',1.333,cosine),ct)),2.)
    rf=mathnode('MULTIPLY',.5,mathnode('ADD',rs,rp))
    attr=nodes.new('ShaderNodeAttribute');attr.attribute_name='thickness_nm'
    rgb=nodes.new('ShaderNodeCombineColor');rgb.mode='RGB'
    for k,wl in enumerate([650.,530.,450.]):
        phase=mathnode('MULTIPLY',2*math.pi*1.333/wl,mathnode('MULTIPLY',attr.outputs['Fac'],ct))
        sine=mathnode('POWER',mathnode('SINE',phase),2.)
        numerator=mathnode('MULTIPLY',mathnode('MULTIPLY',4.,rf),sine)
        reflectance=mathnode('DIVIDE',numerator,mathnode('ADD',mathnode('POWER',mathnode('SUBTRACT',1.,rf),2.),numerator))
        links.new(reflectance,rgb.inputs[k])
    gloss=nodes.new('ShaderNodeBsdfAnisotropic');gloss.inputs['Roughness'].default_value=.045;links.new(rgb.outputs[0],gloss.inputs['Color'])
    transmission=nodes.new('ShaderNodeMixRGB');transmission.blend_type='SUBTRACT';transmission.inputs[0].default_value=1.;transmission.inputs[1].default_value=(1,1,1,1);links.new(rgb.outputs[0],transmission.inputs[2])
    transparent=nodes.new('ShaderNodeBsdfTransparent');links.new(transmission.outputs[0],transparent.inputs[0])
    add=nodes.new('ShaderNodeAddShader');links.new(gloss.outputs[0],add.inputs[0]);links.new(transparent.outputs[0],add.inputs[1])
    out=nodes.new('ShaderNodeOutputMaterial');links.new(add.outputs[0],out.inputs['Surface'])
    return m


def finish(o,name,mat):
    o.name=name;o.data.materials.append(mat)
    if o.type=='MESH':
        for f in o.data.polygons:f.use_smooth=True
    return o


def disk(name,p,r,depth,mat):
    bpy.ops.mesh.primitive_cylinder_add(vertices=128,radius=r*S,depth=depth*S,location=np.array(p)*S)
    o=finish(bpy.context.object,name,mat);m=o.modifiers.new('Machined edge','BEVEL');m.width=.0007*S;m.segments=3;return o


def ring(name,p,r,tube,mat,vertical=False):
    bpy.ops.mesh.primitive_torus_add(major_segments=128,minor_segments=12,major_radius=r*S,minor_radius=tube*S,location=np.array(p)*S,rotation=(math.pi/2 if vertical else 0,0,0))
    return finish(bpy.context.object,name,mat)


def area(name,p,target,power,size,size_y,color):
    d=bpy.data.lights.new(name,'AREA');d.energy=power;d.shape='RECTANGLE';d.size=size*S;d.size_y=size_y*S;d.color=color
    o=bpy.data.objects.new(name,d);bpy.context.collection.objects.link(o);o.location=np.array(p)*S;o.rotation_euler=(Vector(np.array(target)*S)-o.location).to_track_quat('-Z','Y').to_euler()


def studio_environment(scene):
    # Smooth neutral illumination reveals spectral film color without rectangular
    # reflected card edges. This static radiance map is not a material texture.
    h,w=512,1024;yy,xx=np.mgrid[:h,:w];theta=np.pi*(yy+.5)/h;phi=2*np.pi*(xx+.5)/w
    direction=np.stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)],-1)
    radiance=np.full((h,w),.18)
    for vector,power,width in [((-.7,-.6,.5),7.,.52),((.8,.3,.6),5.,.42),((.1,.7,.2),3.,.65)]:
        a=np.array(vector);a/=np.linalg.norm(a);radiance+=power*np.exp((direction@a-1)/width**2)
    rgba=np.concatenate([np.repeat(radiance[...,None],3,axis=2),np.ones((h,w,1))],2).astype(np.float32)
    image=bpy.data.images.new('Procedural neutral studio radiance',width=w,height=h,float_buffer=True)
    image.pixels.foreach_set(rgba.ravel());image.update()
    scene.world.use_nodes=True;nodes=scene.world.node_tree.nodes;nodes.clear();links=scene.world.node_tree.links
    env=nodes.new('ShaderNodeTexEnvironment');env.image=image
    light=nodes.new('ShaderNodeBackground');links.new(env.outputs['Color'],light.inputs[0])
    background=nodes.new('ShaderNodeBackground');background.inputs[0].default_value=(.008,.012,.018,1)
    ray=nodes.new('ShaderNodeLightPath');mix=nodes.new('ShaderNodeMixShader')
    links.new(ray.outputs['Is Camera Ray'],mix.inputs[0]);links.new(light.outputs[0],mix.inputs[1]);links.new(background.outputs[0],mix.inputs[2])
    out=nodes.new('ShaderNodeOutputWorld');links.new(mix.outputs[0],out.inputs[0])


def setup(cfg,args):
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.samples=args.samples;scene.cycles.use_denoising=True;scene.cycles.device='GPU';scene.cycles.seed=29;scene.render.use_persistent_data=True
    prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices()
    for device in prefs.devices:device.use=device.type=='OPTIX'
    scene.render.resolution_x=args.width;scene.render.resolution_y=args.height;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.cycles.transparent_max_bounces=24;scene.cycles.max_bounces=12
    scene.view_settings.view_transform='AgX';scene.view_settings.look='AgX - Medium High Contrast';scene.view_settings.exposure=-.6
    studio_environment(scene)
    graphite=material('Satin graphite',(.025,.036,.051),.7,.24);gold=material('Champagne anodized metal',(.5,.31,.13),.78,.19);ceramic=material('Porcelain',(.72,.76,.80),.05,.23)
    floor=material('Midnight cyclorama',(.0015,.0025,.0045),0.,.85)
    bpy.ops.mesh.primitive_plane_add(size=2000,location=(0,0,-.066*S));ground=finish(bpy.context.object,'Studio floor',floor)
    style=args.style or cfg['scene'];duet=style in ['bubble_duet','bubble_collision'];giant=style=='giant_recoil'
    radius=.105 if style=='bubble_collision' else (.095 if duet else (.078 if giant else .062))
    base=-.074 if giant else -.061
    if not giant:
        disk('Exhibition plinth',(0,0,base),radius,.008,graphite);disk('Inset polished top',(0,0,base+.0045),radius*.97,.002,graphite)
        ring('Fine brass perimeter',(0,0,base+.0056),radius*.94,.00045,gold)
    else:
        ground.visible_glossy=False
    if not duet and not giant:
        ring('Suspended halo',(0,.045,-.005),.050,.0007,gold,True)
    elif duet:
        for b in cfg['bubbles']:ring('Assembly guide',(b['center'][0],b['center'][1],-.0548),b['radius']*.975,.00035,gold)
    if style=='bubble_collision':
        # Keep the apparatus visible to the camera while allowing continuous
        # studio illumination to reveal the actual film curvature.
        for obj in scene.objects:
            if obj.type=='MESH':obj.visible_glossy=False
    if cfg['scene']=='driven_bubble':
        for side in [-1,1]:
            ring('Air nozzle',(side*.061,0,0),.009,.0012,graphite,True)
    # Broad softboxes produce continuous interference colors and readable curvature.
    bpy.ops.object.camera_add();camera=bpy.context.object;camera.name='Orbit camera';camera.data.lens=50;scene.camera=camera
    return scene,camera,soap()


def main():
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--indices');p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=720);p.add_argument('--samples',type=int,default=48);p.add_argument('--resume',action='store_true');p.add_argument('--style',choices=['giant_recoil','bubble_collision'])
    a=p.parse_args(argv);cfg=json.loads((a.input/'config.json').read_text());a.output.mkdir(parents=True,exist_ok=True);scene,camera,film=setup(cfg,a)
    files=sorted(a.input.glob('[0-9]*.npz'));indices=[int(i) for i in a.indices.split(',')] if a.indices else [int(f.stem) for f in files]
    bubble=None
    for i in indices:
        target=a.output/f'{i:05d}.png'
        if a.resume and target.exists():continue
        data=np.load(a.input/f'{i:05d}.npz');verts=data['vertices']*S
        if bubble is None:
            mesh=bpy.data.meshes.new('Recorded thin film');mesh.from_pydata(verts.tolist(),[],data['faces'].tolist());mesh.update()
            bubble=bpy.data.objects.new('MELP reconstructed surface',mesh);bpy.context.collection.objects.link(bubble);finish(bubble,bubble.name,film)
            mesh.attributes.new('thickness_nm','FLOAT','POINT')
        else:bubble.data.vertices.foreach_set('co',verts.ravel())
        bubble.data.attributes['thickness_nm'].data.foreach_set('value',data['thickness_nm']);bubble.data.update()
        f=i/max(1,cfg.get('camera_frames',cfg['frames'])-1);angle=math.radians(-99+32*f);radius=.39 if cfg['scene']=='bubble_duet' else .315
        target_center=np.array([0.,0.,-.012])
        if (a.style or cfg['scene'])=='giant_recoil':
            radius=.33;angle=math.radians(-96+30*f);target_center=data['vertices'].mean(0)*np.array([.8,.8,.5])+np.array([0,0,-.009])
        if (a.style or cfg['scene'])=='bubble_collision':
            angle=math.radians(-96+48*f);target_center=data['vertices'].mean(0)*np.array([.7,.7,.5])+np.array([0,0,-.012])
            screen_right=np.array([-math.sin(angle),math.cos(angle),0.])
            half_width=np.max(np.abs((data['vertices']-target_center)@screen_right))
            radius=max(.42-.05*f,3.25*half_width)
        camera.location=np.array([radius*math.cos(angle),radius*math.sin(angle),.075+.012*math.sin(math.pi*f)])*S
        camera.rotation_euler=(Vector(target_center*S)-camera.location).to_track_quat('-Z','Y').to_euler()
        scene.render.filepath=str(target.resolve());start=time.monotonic();bpy.ops.render.render(write_still=True)
        print(json.dumps(dict(frame=i,seconds=time.monotonic()-start)),flush=True)
    name=f'render_{indices[0]:05d}_{indices[-1]:05d}.json'
    (a.output/name).write_text(json.dumps(dict(indices=indices,width=a.width,height=a.height,samples=a.samples,input=str(a.input.resolve()),style=a.style,
        renderer_sha256=RENDERER_SHA256,shader='Three-band air/water/air thin-film Fresnel interference from simulated thickness'),indent=2)+'\n')

if __name__=='__main__':main()
