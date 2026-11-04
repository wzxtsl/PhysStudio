"""Three-dimensional thin-film presets; SI physics with explicit slow-motion playback."""

def make(scene='vortex_bubble'):
    cfg=dict(scene=scene,seed=29,e_count=1622,l_count=26000,support=2.5,rho=997.,rt=8.3144598*298.15,
             sigma0=.07275,sigma=.03,atmosphere=101325.,thickness=600.e-9,dt=.0005,frame_dt=.005,
             fps=30,frames=241,normal_damping=40.,tangent_damping=.03,redistribute=.0002,affine_strength=1.0,
             rotation_speed=.09,vortex_speed=.20,bubbles=[dict(radius=.04,center=[0,0,0])])
    if scene=='vortex_bubble':return cfg
    if scene=='giant_recoil':
        cfg.update(support=3.2,atmosphere=100.,normal_added_mass=.06,normal_damping=2.,
                   redistribute=.01,dt=.00025,frame_dt=.0025,rotation_speed=.1,vortex_speed=.2,
                   neighbor_capacity=256,redistribution_iterations=2,initial_relaxation_steps=20,
                   bubbles=[dict(radius=.04,center=[0,0,0],axes=[1.65,.72,.84])])
        return cfg
    if scene=='driven_bubble':
        cfg.update(rotation_speed=.015,vortex_speed=.03,pressure_amplitude=.55,support=3.,
                   atmosphere=100.,normal_added_mass=.008,redistribute=.001)
        return cfg
    if scene=='bubble_duet':
        cfg.update(rotation_speed=.025,vortex_speed=.06,atmosphere=100.,normal_added_mass=.008,
                   normal_damping=25.,contact_stiffness=8000.,redistribute=.001,
                   bubbles=[dict(radius=.04,center=[-.041,0,0],velocity=[.014,0,0]),dict(radius=.04,center=[.041,0,0],velocity=[-.014,0,0])])
        return cfg
    if scene=='bubble_collision':
        cfg=make('bubble_duet')
        cfg.update(scene=scene,support=3.2,normal_damping=1.5,redistribute=.008,dt=.00025,
                   frame_dt=.0025,preserve_contact_traction=1,contact_stiffness=50000.,
                   multi_region=False,redistribution_iterations=2,neighbor_capacity=256,
                   normal_added_mass=.06,normal_smoothing=.35,normal_smoothing_passes=1,
                   contact_redistribution=False,continuous_contact=True,initial_relaxation_steps=20,
                   tension_sharing=False,contact_adhesion=False,
                   bubbles=[dict(radius=.04,axes=[1.35,.78,.95],center=[s*.056,s*.006,0],
                                 velocity=[-s*.45,0,0]) for s in [-1,1]])
        return cfg
    raise ValueError(scene)
