"""Dynamic plant graph with GPU signals and resource-limited meristem extension."""
import math
import numpy as np
import warp as wp
from . import kernels as k


def normalized(x):
    x = np.asarray(x, dtype=float)
    return x / max(np.linalg.norm(x), 1.0e-12)


class Plant:
    def __init__(self, cfg, soil):
        self.cfg = cfg
        self.soil = soil
        self.device = soil.device
        self.rng = np.random.default_rng(cfg["seed"])
        self.nodes = []
        self.history = []
        self.branch_counter = 0
        for kind in [0, 1]:
            b = self.add(
                [0, 0, -0.001 * kind], -1, kind, 0, [0, 0, 1 - 2 * kind], 0, 0, 0
            )
            self.add(
                [0, 0, 0.06 * (1 - 2 * kind)], b, kind, 0, [0, 0, 1 - 2 * kind], 0, 0, 1
            )

    def add(self, p, parent, kind, order, direction, birth, branch, progress):
        self.nodes.append(
            dict(
                p=np.array(p, dtype=float),
                parent=parent,
                kind=kind,
                order=order,
                direction=normalized(direction),
                birth=birth,
                branch=branch,
                progress=progress,
                tip=True,
                radius=0.001,
                vigor=0.0,
                light=1.0,
                water=0.72,
                nutrient=0.65,
            )
        )
        if parent >= 0:
            self.nodes[parent]["tip"] = False
        return len(self.nodes) - 1

    def arrays(self, step):
        n = self.nodes
        a = {
            key: np.array(
                [v[key] for v in n],
                dtype=np.float32 if key in ["p", "direction", "birth"] else np.int32,
            )
            for key in ["p", "direction", "parent", "kind", "order", "tip", "birth"]
        }
        a["age"] = step - a["birth"]
        return a

    def wp_array(self, a, dtype):
        return wp.array(a, dtype=dtype, device=self.device)

    def step(self, step):
        cfg = self.cfg
        s = self.soil
        s.transport()
        a = self.arrays(step)
        count = len(self.nodes)
        p = self.wp_array(a["p"], wp.vec3)
        parent = self.wp_array(a["parent"], int)
        kind = self.wp_array(a["kind"], int)
        order = self.wp_array(a["order"], int)
        tip = self.wp_array(a["tip"], int)
        age = self.wp_array(a["age"], float)
        light = wp.empty(count, dtype=float, device=self.device)
        soil_values = wp.empty(count, dtype=wp.vec3, device=self.device)
        optimal = wp.empty(count, dtype=wp.vec3, device=self.device)
        wp.launch(
            k.sense,
            count,
            [
                p,
                kind,
                order,
                age,
                s.water,
                s.nutrients,
                s.resistance,
                *s.geometry,
                cfg["light_scale"],
                light,
                soil_values,
                optimal,
            ],
            device=self.device,
        )
        actual = s.absorb(p, parent, kind, order)
        l = light.numpy()
        sv = soil_values.numpy()
        tips = a["tip"].astype(bool)
        shoot = a["kind"] == 0
        root = ~shoot
        # Fluxes are converted to abstract developmental vigor, as in the paper.
        # Seed reserves support establishment before the first fine roots and leaves.
        seed = 3.5 * math.exp(-step / 22.0)
        sf = seed + float((l * shoot * np.where(a["order"] > 0, 0.36, 0.025)).sum())
        rf = seed + float(actual.sum()) / cfg["uptake"] * 0.32
        total = min(sf, rf)
        shoot_count = max(1, int((tips & shoot).sum()))
        root_count = max(1, int((tips & root).sum()))
        maintenance = 0.002 * count
        growth = max(0.0, total - maintenance)
        # Demand-weighted allocation retains both systems; counts enter the official auto-balance rule.
        sm = float(shoot_count)
        rm = float(root_count)
        shoot_weight = sm * math.sqrt(
            max(1, int(root.sum())) / max(1, int(shoot.sum()))
        )
        root_weight = rm * math.sqrt(max(1, int(shoot.sum())) / max(1, int(root.sum())))
        shoot_budget = growth * shoot_weight / (shoot_weight + root_weight)
        root_budget = growth - shoot_budget
        weights = wp.zeros(count, dtype=float, device=self.device)
        wp.launch(
            k.accumulate,
            count,
            [tip, parent, kind, order, soil_values, light, weights, 0.995],
            device=self.device,
        )
        weights_np = weights.numpy()
        own = np.where(
            root,
            np.maximum(0.02, sv[:, 1]) / np.maximum(1.0, sv[:, 2]),
            np.maximum(0.02, l),
        )
        own *= np.power(0.82, a["order"]) * tips
        sums = own.copy()
        for i, node in enumerate(self.nodes):
            par = node["parent"]
            if par >= 0:
                if node["order"] == self.nodes[par]["order"]:
                    weights_np[i] *= (
                        cfg["root_apical"] if node["kind"] else cfg["shoot_apical"]
                    )
                sums[par] += weights_np[i]
        weights = self.wp_array(weights_np, float)
        siblings = np.array([sums[max(0, n["parent"])] for n in self.nodes], np.float32)
        fraction = own / np.maximum(sums, 1.0e-12)
        nxt = wp.empty(count, dtype=wp.vec3, device=self.device)
        direction = wp.empty_like(nxt)
        alloc = wp.empty(count, dtype=float, device=self.device)
        wp.launch(
            k.elongate,
            count,
            [
                p,
                self.wp_array(a["direction"], wp.vec3),
                kind,
                order,
                tip,
                parent,
                weights,
                self.wp_array(siblings, float),
                self.wp_array(fraction.astype(np.float32), float),
                optimal,
                s.nutrients,
                s.resistance,
                s.sdf,
                *s.geometry,
                cfg["seed"],
                step,
                cfg["root_style"],
                cfg["shoot_length"],
                cfg["root_length"],
                shoot_budget,
                root_budget,
                nxt,
                direction,
                alloc,
            ],
            device=self.device,
        )
        q = nxt.numpy()
        d = direction.numpy()
        v = alloc.numpy()
        for i, node in enumerate(self.nodes):
            node["vigor"] = float(v[i])
            node["light"] = float(l[i])
            node["water"] = float(sv[i, 0])
            node["nutrient"] = float(sv[i, 1])
        for i in np.flatnonzero(tips):
            node = self.nodes[i]
            typ = node["kind"]
            o = node["order"]
            progress = node["progress"]
            limits = cfg["root_limits"] if typ else cfg["shoot_limits"]
            if progress >= limits[o]:
                node["tip"] = False
                continue
            if np.linalg.norm(q[i] - node["p"]) < 0.001:
                continue
            child = self.add(q[i], i, typ, o, d[i], step, node["branch"], progress + 1)
            interval = (3 if o == 0 else 5) if typ else (5 if o < 2 else 4)
            eligible = progress % interval == 0 and o < len(limits) - 1
            if not typ and o == 0 and node["p"][2] < 0.45:
                eligible = False
            probability = (
                cfg["root_branch_probability"]
                if typ
                else cfg["shoot_branch_probability"]
            )
            if typ and o == 0 and progress <= 12:
                probability = 0.98
            if typ:
                probability *= min(1.35, 0.6 + sv[i, 1] * 0.6)
            probability *= min(1.0, v[i] / 0.45)
            if eligible and self.rng.random() < probability:
                angle = math.radians(
                    (67 if typ else cfg["shoot_branch_angle"]) + self.rng.normal(0, 8)
                )
                roll = progress * 2.399963 + node["branch"] * 1.7
                axis = normalized(d[i])
                u = normalized(np.cross(axis, [0, 1, 0]))
                vv = normalized(np.cross(axis, u))
                lateral = normalized(
                    math.cos(angle) * axis
                    + math.sin(angle) * (math.cos(roll) * u + math.sin(roll) * vv)
                )
                if typ:
                    lateral[2] = -abs(lateral[2])
                    if o == 0:
                        lateral = normalized(
                            [
                                math.cos(roll),
                                math.sin(roll),
                                -0.24 if cfg["root_style"] == 2 else -0.48,
                            ]
                        )
                    elif cfg["root_style"] == 2 and o == 1 and self.rng.random() < 0.38:
                        lateral = normalized([lateral[0] * 0.3, lateral[1] * 0.3, -1])
                else:
                    lateral[2] = max(0.13, lateral[2])
                    lateral = normalized(lateral)
                self.branch_counter += 1
                self.add(q[i], child, typ, o + 1, lateral, step, self.branch_counter, 0)
                # A lateral bud does not deactivate its parent's apical meristem.
                self.nodes[child]["tip"] = True
        self.update_radius(step)
        record = dict(
            step=step,
            nodes=len(self.nodes),
            shoot_nodes=int(shoot.sum()),
            root_nodes=int(root.sum()),
            shoot_tips=shoot_count,
            root_tips=root_count,
            shoot_flux=sf,
            root_flux=rf,
            vigor=total,
            maintenance=maintenance,
            shoot_budget=shoot_budget,
            root_budget=root_budget,
            shoot_allocated=float(v[shoot].sum()),
            root_allocated=float(v[root].sum()),
            shoot_height=float(a["p"][shoot, 2].max()),
            root_depth=float(-a["p"][root, 2].min()),
        )
        record.update(s.audit())
        self.history.append(record)
        return record

    def update_radius(self, step):
        pipe = np.ones(len(self.nodes)) * 0.05
        for i in reversed(range(len(self.nodes))):
            n = self.nodes[i]
            if n["tip"]:
                pipe[i] += 1.0
            if n["parent"] >= 0:
                pipe[n["parent"]] += pipe[i]
        for i, n in enumerate(self.nodes):
            r = (
                self.cfg["root_tip_radius"]
                if n["kind"]
                else self.cfg["shoot_tip_radius"]
            )
            n["radius"] = float(
                r * pipe[i] ** 0.52
                + 0.00016 * max(0, step - n["birth"]) * 0.55 ** n["order"]
            )

    def save(self, path, step):
        a = self.arrays(step)
        for key in [
            "radius",
            "vigor",
            "light",
            "water",
            "nutrient",
            "progress",
            "branch",
        ]:
            a[key] = np.array([n[key] for n in self.nodes])
        np.savez_compressed(path, **a)
