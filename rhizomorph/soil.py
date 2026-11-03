"""Conservative soil water/solute transport on a regular Warp grid."""
import numpy as np
import warp as wp
from . import kernels as k


class Soil:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.nx, self.ny, self.nz = cfg["grid"]
        self.count = self.nx * self.ny * self.nz
        self.lo = np.array(cfg["bounds"][0])
        self.h = (np.array(cfg["bounds"][1]) - self.lo) / cfg["grid"]
        z, y, x = np.meshgrid(
            np.arange(self.nz), np.arange(self.ny), np.arange(self.nx), indexing="ij"
        )
        self.xyz = self.lo + (np.stack([x, y, z], -1) + 0.5) * self.h
        p = self.xyz
        capacity = np.full(x.shape, cfg["capacity"])
        perm = np.full(x.shape, cfg["permeability"])
        resistance = np.full(x.shape, cfg["resistance"])
        water = np.full(x.shape, cfg["moisture"]) * cfg["water_scale"]
        nutrients = np.full(x.shape, cfg["nutrient"])
        sdf = np.full(x.shape, 20.0)
        # Spatially smooth heterogeneity avoids a regular lattice of roots.
        nutrients *= 1.0 + 0.10 * np.sin(1.7 * p[..., 0] + 0.4) * np.cos(
            1.5 * p[..., 1]
        )
        for patch in cfg["patches"]:
            d = np.linalg.norm(p - np.array(patch["center"]), axis=-1)
            nutrients += patch["strength"] * np.exp(-0.5 * (d / patch["radius"]) ** 2)
        if cfg["hardpan"] is not None:
            layer = 1.0 / (
                1.0 + np.exp(np.clip((p[..., 2] - cfg["hardpan"]) / 0.10, -60, 60))
            )
            resistance += 18.0 * layer
            perm *= 1.0 - 0.96 * layer
            capacity += 0.4 * layer
        for rock in cfg["rocks"]:
            r = np.array(rock["radii"])
            d = (
                np.linalg.norm((p - np.array(rock["center"])) / r, axis=-1) - 1.0
            ) * min(r)
            sdf = np.minimum(sdf, d)
        inside = sdf < 0.0
        perm[inside] = 0.0
        water[inside] = 0.0
        nutrients[inside] = 0.0
        resistance[inside] = 1.0e4
        self.static = dict(
            capacity=capacity, permeability=perm, resistance=resistance, sdf=sdf
        )
        for name, value in [
            ("water", water),
            ("nutrients", nutrients),
            *self.static.items(),
        ]:
            setattr(
                self,
                name,
                wp.array(value.astype(np.float32).ravel(), dtype=float, device=device),
            )
        self.water_tmp = wp.empty(self.count, dtype=float, device=device)
        self.nutrient_tmp = wp.empty_like(self.water_tmp)
        self.fw = wp.empty(self.count, dtype=wp.vec3, device=device)
        self.fn = wp.empty_like(self.fw)
        self.demand = wp.zeros_like(self.water)
        self.ledger = wp.zeros(4, dtype=wp.float64, device=device)
        self.initial_water = float(water.sum())
        self.initial_nutrients = float(nutrients.sum())

    @property
    def geometry(self):
        return [wp.vec3(*self.lo), wp.vec3(*self.h), self.nx, self.ny, self.nz]

    def transport(self, rain=None, drain=0.0012):
        c = self.cfg
        for _ in range(c["soil_substeps"]):
            wp.launch(
                k.face_flux,
                self.count,
                [
                    self.water,
                    self.nutrients,
                    self.capacity,
                    self.permeability,
                    self.nx,
                    self.ny,
                    self.nz,
                    wp.vec3(*self.h),
                    c["soil_dt"],
                    c["diffusion"],
                    c["gravity"],
                    self.fw,
                    self.fn,
                ],
                device=self.device,
            )
            wp.launch(
                k.divergence,
                self.count,
                [
                    self.water,
                    self.nutrients,
                    self.fw,
                    self.fn,
                    self.nx,
                    self.ny,
                    self.nz,
                    self.water_tmp,
                    self.nutrient_tmp,
                ],
                device=self.device,
            )
            self.water, self.water_tmp = self.water_tmp, self.water
            self.nutrients, self.nutrient_tmp = self.nutrient_tmp, self.nutrients
        wp.launch(
            k.boundary,
            self.count,
            [
                self.water,
                self.nutrients,
                self.permeability,
                self.nx,
                self.ny,
                self.nz,
                c["rain"] * c["water_scale"] if rain is None else rain,
                drain,
                self.ledger,
            ],
            device=self.device,
        )

    def absorb(self, p, parent, kind, order):
        self.demand.zero_()
        n = len(p)
        cell = wp.empty(n, dtype=int, device=self.device)
        request = wp.empty(n, dtype=float, device=self.device)
        actual = wp.empty_like(request)
        wp.launch(
            k.uptake_request,
            n,
            [
                p,
                parent,
                kind,
                order,
                self.water,
                self.capacity,
                *self.geometry,
                self.cfg["uptake"],
                self.demand,
                cell,
                request,
            ],
            device=self.device,
        )
        wp.launch(
            k.uptake_actual,
            n,
            [request, cell, self.demand, self.water, actual],
            device=self.device,
        )
        wp.launch(
            k.uptake_apply,
            self.count,
            [self.water, self.demand, self.ledger],
            device=self.device,
        )
        return actual.numpy()

    def audit(self):
        w = self.water.numpy()
        nu = self.nutrients.numpy()
        l = self.ledger.numpy().astype(float)
        expected = self.initial_water + l[0] - l[1] - l[2]
        return dict(
            water=float(w.sum(dtype=np.float64)),
            water_balance_relative_error=float(
                (w.sum(dtype=np.float64) - expected) / self.initial_water
            ),
            nutrient_balance_relative_error=float(
                (nu.sum(dtype=np.float64) + l[3] - self.initial_nutrients)
                / self.initial_nutrients
            ),
            water_min=float(w.min()),
            nutrient_min=float(nu.min()),
            rain=float(l[0]),
            drain=float(l[1]),
            uptake=float(l[2]),
        )
