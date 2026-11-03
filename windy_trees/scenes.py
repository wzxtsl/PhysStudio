"""Mature Rhizomorph trees exposed to three wind histories."""
from copy import deepcopy

SCENES = ('oak_taproot', 'maple_heart', 'birch_lateral')
BASE = dict(seconds=10.0, fps=30, substeps=8, solver_iterations=60, young=1.0e9,
            wood_density=650.0, leaf_areal_mass=0.09, damping_ratio=0.18,
            air_density=1.225, spacing=0.38, support=0.80, sound_speed=15.0,
            viscosity=0.035, bounds=[[-6.0,-4.4,0.02],[7.5,4.4,7.2]],
            wind_speed=7.5, turbulence=1.2, direction=0.0, turn=0.0,
            drag_wood=0.8, drag_leaf=0.7, leaf_stiffness=0.0060,
            leaf_damping=0.28, seed=64, warmup=0.6,
            camera_start=-110.0, camera_orbit=36.0, width=1280, height=960,
            samples=24, coupling=True, profile='gusts',trace_count=4000)

def scene_config(name):
    if name not in SCENES:
        raise ValueError(name)
    cfg=deepcopy(BASE);cfg['scene']=name
    if name=='oak_taproot':
        cfg.update(title='Oak / Passing gusts',wind_speed=8.5,young=1.0e9)
    elif name=='maple_heart':
        cfg.update(title='Maple / Turning wind',wind_speed=7.0,turn=0.75,
                   young=1.3e9,profile='turning',camera_orbit=44.)
    else:
        cfg.update(title='Birch / Turbulent breeze',wind_speed=9.0,
                   turbulence=1.8,young=1.1e9,leaf_stiffness=0.0045,
                   direction=-.15,profile='turbulent',camera_orbit=40.)
    return cfg
