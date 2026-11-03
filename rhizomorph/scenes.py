"""Reproducible developmental scenes; distances are in meters."""
from copy import deepcopy

BASE = dict(
    steps=180,
    seed=27,
    grid=[64, 64, 40],
    bounds=[[-3.6, -3.6, -4.5], [3.6, 3.6, 0.0]],
    soil_dt=0.025,
    soil_substeps=8,
    diffusion=0.24,
    gravity=0.045,
    capacity=1.0,
    permeability=0.11,
    resistance=1.0,
    moisture=0.72,
    nutrient=0.65,
    water_scale=1.0,
    light_scale=1.0,
    uptake=0.000025,
    shoot_length=0.047,
    root_length=0.058,
    root_style=0,
    shoot_branch_angle=54.0,
    shoot_branch_probability=0.62,
    root_branch_probability=0.68,
    shoot_apical=1.25,
    root_apical=1.10,
    shoot_tip_radius=0.0045,
    root_tip_radius=0.0034,
    shoot_limits=[92, 38, 18, 7],
    root_limits=[58, 34, 19, 9, 4],
    leaf_size=0.185,
    leaf_shape="oak",
    leaf_color=[0.055, 0.20, 0.014],
    rocks=[],
    patches=[],
    hardpan=None,
    rain=0.0008,
    camera_start=-112.0,
    camera_orbit=52.0,
    frames=240,
    fps=30,
    title="Oak | Deep taproot",
)


def scene_config(name):
    cfg = deepcopy(BASE)
    cfg["scene"] = name
    if name == "oak_taproot":
        cfg.update(title="Oak / Deep taproot", seed=27, root_style=0)
        cfg["patches"] = [dict(center=[0.4, 0.2, -2.6], radius=1.1, strength=0.8)]
    elif name == "maple_heart":
        cfg.update(
            title="Maple / Rock and resource pockets",
            seed=46,
            root_style=1,
            shoot_length=0.041,
            shoot_branch_angle=62.0,
            shoot_branch_probability=0.70,
            leaf_shape="maple",
            leaf_size=0.19,
            leaf_color=[0.10, 0.25, 0.02],
            shoot_limits=[100, 48, 24, 8],
            root_limits=[40, 42, 24, 10, 4],
        )
        cfg["rocks"] = [
            dict(center=[0.4, 0.05, -1.3], radii=[0.74, 0.62, 0.60]),
            dict(center=[-1.45, 0.35, -2.2], radii=[0.46, 0.56, 0.60]),
        ]
        cfg["patches"] = [
            dict(center=[1.65, 0.15, -1.9], radius=0.65, strength=2.0),
            dict(center=[-0.95, -0.7, -2.7], radius=0.7, strength=1.3),
        ]
    elif name == "birch_lateral":
        cfg.update(
            title="Birch / Shallow roots and sinkers",
            seed=81,
            root_style=2,
            shoot_length=0.048,
            shoot_branch_angle=67.0,
            shoot_apical=1.5,
            leaf_shape="birch",
            leaf_size=0.165,
            leaf_color=[0.10, 0.26, 0.025],
            root_limits=[16, 42, 24, 8, 4],
            shoot_limits=[112, 32, 15, 7],
            hardpan=-1.5,
            permeability=0.06,
        )
        cfg["patches"] = [dict(center=[1.7, -0.6, -0.6], radius=0.85, strength=1.1)]
    else:
        raise ValueError(name)
    return cfg


SCENES = ("oak_taproot", "maple_heart", "birch_lateral")
