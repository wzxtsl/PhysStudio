"""Three 3D mixture experiments, including the paper's tea-brewing sequence."""
import copy
import numpy as np

PRESETS = {
    "tea": dict(
        title="Tea, diffusion & a glass stirrer",
        dx=0.0225,
        dt=0.0025,
        seconds=10.0,
        fps=24,
        bounds=[[-0.45, -0.45, 0.0], [0.45, 0.45, 0.82]],
        cylindrical=True,
        fluid_height=0.56,
        densities=[1000.0, 1050.0, 1030.0],
        drag=0.97,
        diffusion=0.006,
        viscosity=0.002,
        tea_bag=True,
        bag_half=[0.105, 0.036, 0.135],
        bag_initial=[-0.075, 0.0, 0.365],
        rod_radius=0.026,
        rod_half=0.55,
        camera_target=[0.0, 0.0, 0.46],
        bag_lift=0.64,
        rod_start_height=1.75,
        rod_lower_distance=1.13,
        colors=[[0.72, 0.9, 0.96], [0.32, 0.095, 0.014], [0.9, 0.85, 0.8]],
    ),
    "separation": dict(
        title="Density inversion",
        dx=0.032,
        dt=0.0025,
        seconds=5.0,
        fps=24,
        bounds=[[-0.60, -0.34, 0.0], [0.60, 0.34, 0.90]],
        fluid_height=0.65,
        densities=[850.0, 1100.0, 1600.0],
        drag=0.55,
        diffusion=0.00003,
        viscosity=0.0004,
        colors=[[0.91, 0.34, 0.12], [0.91, 0.75, 0.40], [0.055, 0.36, 0.52]],
        camera_target=[0.0, 0.0, 0.36],
    ),
    "collision": dict(
        title="Chromatic collision",
        dx=0.018,
        dt=0.0025,
        seconds=5.0,
        fps=24,
        bounds=[[-0.70, -0.42, 0.0], [0.70, 0.42, 1.0]],
        fluid_height=0.65,
        densities=[900.0, 1200.0, 1000.0],
        drag=0.72,
        diffusion=0.0004,
        viscosity=0.0006,
        colors=[[0.08, 0.56, 0.62], [0.85, 0.27, 0.15], [0.78, 0.65, 0.91]],
        camera_target=[0.0, 0.0, 0.32],
    ),
}


def configuration(name, dx=None):
    cfg = copy.deepcopy(PRESETS[name])
    cfg["scene"] = name
    if dx is not None:
        cfg["dx"] = float(dx)
    if cfg.get("tea_bag"):
        counts = np.maximum(
            1, np.round(2 * np.array(cfg["bag_half"]) / cfg["dx"]).astype(int)
        )
        cfg["bag_half"] = (counts * cfg["dx"] / 2).tolist()
    return cfg


def lattice(lo, hi, dx):
    return np.stack(
        np.meshgrid(
            *[np.arange(a + dx / 2, b, dx) for a, b in zip(lo, hi)], indexing="ij"
        ),
        -1,
    ).reshape(-1, 3)


def poses(cfg, t):
    bag = np.array(cfg.get("bag_initial", [0, 0, 10]), float)
    rod = np.array([0.23, 0.0, cfg.get("rod_start_height", 1.30)])
    if cfg["scene"] == "tea":
        lift = np.clip((t - 3.2) / 1.4, 0.0, 1.0)
        lift = lift * lift * (3 - 2 * lift)
        bag[2] += cfg.get("bag_lift", 0.85) * lift
        bag[0] -= 0.13 * lift
        lower = np.clip((t - 4.7) / 0.8, 0.0, 1.0) if cfg.get("stirring", True) else 0.0
        lower = lower * lower * (3 - 2 * lower)
        rod[2] -= cfg.get("rod_lower_distance", 0.89) * lower
        tau = max(t - 5.5, 0.0) if cfg.get("stirring", True) else 0.0
        theta = 1.8 * (0.5 * tau * tau / 0.7 if tau < 0.7 else tau - 0.35)
        rod[:2] = [0.23 * np.cos(theta), 0.23 * np.sin(theta)]
    return bag, rod


def motion(cfg, t):
    bag, rod = poses(cfg, t)
    bp, rp = poses(cfg, t + 1.0e-4)
    bm, rm = poses(cfg, t - 1.0e-4)
    return dict(
        bag=bag.tolist(),
        bag_velocity=((bp - bm) / 2.0e-4).tolist(),
        rod=rod.tolist(),
        rod_velocity=((rp - rm) / 2.0e-4).tolist(),
    )


def initial_state(cfg):
    dx = cfg["dx"]
    lo, hi = np.array(cfg["bounds"], float)
    height = cfg["fluid_height"]
    allpts = lattice(lo - [3 * dx, 3 * dx, 3 * dx], hi + [3 * dx, 3 * dx, 0], dx)
    if cfg.get("cylindrical"):
        radius = hi[0]
        r = np.linalg.norm(allpts[:, :2], axis=1)
        inside = r < radius
        boundary = (
            (r >= radius) & (r < radius + 2.5 * dx) & (allpts[:, 2] < hi[2])
        ) | ((allpts[:, 2] < 0) & (r < radius + 2.5 * dx))
    else:
        inside = (abs(allpts[:, :2]) < hi[:2]).all(1)
        boundary = (~inside | (allpts[:, 2] < 0)) & (allpts[:, 2] < hi[2])
    fill = inside & (allpts[:, 2] > 0) & (allpts[:, 2] < height)
    if cfg.get("cylindrical"):
        fill &= np.linalg.norm(allpts[:, :2], axis=1) < hi[0] - 0.5 * dx
    bag = np.empty((0, 3))
    baglocal = np.empty((0, 3))
    rod = np.empty((0, 3))
    rodlocal = np.empty((0, 3))
    if cfg.get("tea_bag"):
        center = np.array(cfg["bag_initial"])
        half = np.array(cfg["bag_half"])
        fill &= (abs(allpts - center) > half + dx * 0.55).any(1)
        baglocal = lattice(-half, half, dx)
        bag = baglocal + center
        rr = cfg["rod_radius"]
        rh = cfg["rod_half"]
        rodlocal = lattice([-rr, -rr, -rh], [rr, rr, rh], dx)
        rodlocal = rodlocal[np.linalg.norm(rodlocal[:, :2], axis=1) < rr]
        rod = rodlocal + poses(cfg, 0)[1]
    fluid = allpts[fill]
    if cfg["scene"] == "collision":
        select = (
            (fluid[:, 0] < -0.20)
            & (fluid[:, 0] > -0.59)
            & (abs(fluid[:, 1]) < 0.29)
            & (fluid[:, 2] > 0.05)
            & (fluid[:, 2] < 0.52)
        ) | (
            (fluid[:, 0] > 0.20)
            & (fluid[:, 0] < 0.59)
            & (abs(fluid[:, 1]) < 0.29)
            & (fluid[:, 2] > 0.09)
            & (fluid[:, 2] < 0.63)
        )
        fluid = fluid[select]
    wall = allpts[boundary]
    positions = np.concatenate([fluid, bag, wall, rod]).astype(np.float32)
    kinds = np.concatenate(
        [
            np.zeros(len(fluid)),
            np.full(len(bag), 2),
            np.ones(len(wall)),
            np.full(len(rod), 3),
        ]
    ).astype(np.int32)
    fractions = np.zeros((len(positions), 3), np.float32)
    fractions[:, 0] = 1.0
    vel = np.zeros_like(positions)
    if cfg["scene"] == "tea":
        fractions[len(fluid) : len(fluid) + len(bag)] = [0, 1, 0]
    elif cfg["scene"] == "separation":
        z = fluid[:, 2]
        wav = 0.035 * np.sin(7 * fluid[:, 0]) * np.cos(7 * fluid[:, 1])
        phase = np.where(z < 0.25 + wav, 0, np.where(z < 0.39 + wav, 1, 2))
        fractions[: len(fluid)] = np.eye(3, dtype=np.float32)[phase]
    else:
        phase = np.where(fluid[:, 0] < 0, 0, 1)
        fractions[: len(fluid)] = np.eye(3, dtype=np.float32)[phase]
        vel[: len(fluid), 0] = np.where(fluid[:, 0] < 0, 0.95, -0.70)
    local = positions.copy()
    local[len(fluid) : len(fluid) + len(bag)] = baglocal
    if len(rod):
        local[-len(rod) :] = rodlocal
    return positions, fractions, vel, kinds, local
