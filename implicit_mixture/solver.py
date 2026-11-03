"""Three-phase, volume-incompressible SPH with conservative phase transport.

Equations 2, 16, 19--21 of Xu et al., SIGGRAPH Asia 2023. Pressure
iterations include the phase-mapping mobility. Pairwise common flux limits
preserve both phase volumes and the local partition of unity. Phase momentum
is transported with the donor fluid. All simulation kernels execute in Warp.
"""
import numpy as np
import warp as wp


@wp.func
def row(m: wp.mat33, k: int):
    return wp.vec3(m[k, 0], m[k, 1], m[k, 2])


@wp.func
def W(r: float, h: float):
    q = r / h
    a = float(0.0)
    if q <= 0.5:
        a = 1.0 + 6.0 * (q * q * q - q * q)
    elif q < 1.0:
        a = 2.0 * wp.pow(1.0 - q, 3.0)
    return a * (8.0 / 3.141592653589793) / (h * h * h)


@wp.func
def G(r: wp.vec3, h: float):
    d = wp.length(r)
    q = d / h
    a = float(0.0)
    if d > 1.0e-8 and q < 1.0:
        if q <= 0.5:
            a = 6.0 * (3.0 * q * q - 2.0 * q)
        else:
            a = -6.0 * (1.0 - q) * (1.0 - q)
        return r * (a * (8.0 / 3.141592653589793) / (d * h * h * h * h))
    return wp.vec3(0.0)


@wp.kernel(enable_backward=False)
def mixture(
    u: wp.array(dtype=wp.mat33),
    a: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    rho: wp.array(dtype=float),
    mobility: wp.array(dtype=float),
    density: wp.vec3,
    cd: float,
    nf: int,
):
    i = wp.tid()
    rr = wp.dot(a[i], density)
    vv = wp.vec3(0.0)
    beta = float(0.0)
    for k in range(3):
        vv += a[i][k] * row(u[i], k)
        beta += a[i][k] * (cd + (1.0 - cd) * rr / density[k])
    rho[i] = rr
    v[i] = vv
    mobility[i] = beta / rr
    if i >= nf:
        mobility[i] = 0.0


@wp.kernel(enable_backward=False)
def kinematics(
    x: wp.array(dtype=wp.vec3),
    base: wp.array(dtype=wp.vec3),
    kind: wp.array(dtype=int),
    v: wp.array(dtype=wp.vec3),
    u: wp.array(dtype=wp.mat33),
    bag: wp.vec3,
    bagv: wp.vec3,
    rod: wp.vec3,
    rodv: wp.vec3,
):
    i = wp.tid()
    vel = wp.vec3(0.0)
    if kind[i] == 2:
        x[i] = base[i] + bag
        vel = bagv
    elif kind[i] == 3:
        x[i] = base[i] + rod
        vel = rodv
    if kind[i] > 0:
        v[i] = vel
        u[i] = wp.mat33(
            vel[0], vel[1], vel[2], vel[0], vel[1], vel[2], vel[0], vel[1], vel[2]
        )


@wp.kernel(enable_backward=False)
def neighbor_cache(
    x: wp.array(dtype=wp.vec3),
    grid: wp.uint64,
    h: float,
    vol: float,
    ids: wp.array2d(dtype=int),
    grad: wp.array2d(dtype=wp.vec3),
    count: wp.array(dtype=int),
    psi: wp.array(dtype=float),
    overflow: wp.array(dtype=int),
):
    i = wp.tid()
    n = int(0)
    s = W(0.0, h) * vol
    for j in wp.hash_grid_query(grid, x[i], h):
        if j != i:
            r = x[i] - x[j]
            if wp.length(r) < h:
                if n < ids.shape[1]:
                    ids[i, n] = j
                    grad[i, n] = G(r, h) * vol
                    s += vol * W(wp.length(r), h)
                n += 1
    count[i] = wp.min(n, ids.shape[1])
    psi[i] = s
    wp.atomic_max(overflow, 0, n)


@wp.kernel(enable_backward=False)
def diagonal(
    ids: wp.array2d(dtype=int),
    grad: wp.array2d(dtype=wp.vec3),
    count: wp.array(dtype=int),
    mobility: wp.array(dtype=float),
    diag: wp.array(dtype=float),
):
    i = wp.tid()
    gs = wp.vec3(0.0)
    dd = float(0.0)
    for t in range(count[i]):
        j = ids[i, t]
        g = grad[i, t]
        gs += g
        dd += mobility[j] * wp.dot(g, g)
    diag[i] = mobility[i] * wp.dot(gs, gs) + dd


@wp.kernel(enable_backward=False)
def pressure_residual(
    v: wp.array(dtype=wp.vec3),
    ids: wp.array2d(dtype=int),
    grad: wp.array2d(dtype=wp.vec3),
    count: wp.array(dtype=int),
    psi: wp.array(dtype=float),
    diag: wp.array(dtype=float),
    pressure: wp.array(dtype=float),
    error: wp.array(dtype=float),
    dt: float,
    density_stage: int,
    relaxation: float,
):
    i = wp.tid()
    b = float(0.0)
    for t in range(count[i]):
        b += wp.dot(grad[i, t], v[i] - v[ids[i, t]])
    if density_stage == 1:
        b += (psi[i] - 1.0) / dt
    elif count[i] < 18:
        b = 0.0
    b = wp.max(b, 0.0)
    pressure[i] = relaxation * b / wp.max(diag[i], 1.0e-8)
    wp.atomic_add(error, 0, b * dt / float(pressure.shape[0]))
    wp.atomic_max(error, 1, b * dt)


@wp.kernel(enable_backward=False)
def pressure_update(
    u: wp.array(dtype=wp.mat33),
    rho: wp.array(dtype=float),
    ids: wp.array2d(dtype=int),
    grad: wp.array2d(dtype=wp.vec3),
    count: wp.array(dtype=int),
    p: wp.array(dtype=float),
    density: wp.vec3,
    cd: float,
    nf: int,
):
    i = wp.tid()
    impulse = wp.vec3(0.0)
    for t in range(count[i]):
        j = ids[i, t]
        pj = float(0.0)
        if j < nf:
            pj = p[j]
        impulse -= (p[i] + pj) * grad[i, t]
    m = u[i]
    for k in range(3):
        vk = row(m, k) + (cd / rho[i] + (1.0 - cd) / density[k]) * impulse
        for axis in range(3):
            m[k, axis] = vk[axis]
    u[i] = m


@wp.kernel(enable_backward=False)
def advection_forces(
    x: wp.array(dtype=wp.vec3),
    u: wp.array(dtype=wp.mat33),
    v: wp.array(dtype=wp.vec3),
    ids: wp.array2d(dtype=int),
    grad: wp.array2d(dtype=wp.vec3),
    count: wp.array(dtype=int),
    dt: float,
    h: float,
    cd: float,
    nu: float,
    gravity: wp.vec3,
):
    i = wp.tid()
    m = u[i]
    for k in range(3):
        acc = gravity
        uk = row(m, k)
        for t in range(count[i]):
            j = ids[i, t]
            r = x[i] - x[j]
            dv = cd * (v[i] - v[j]) + (1.0 - cd) * (uk - v[j])
            acc += (
                10.0 * nu * wp.dot(dv, r) / (wp.dot(r, r) + 0.01 * h * h) * grad[i, t]
            )
        vk = uk + dt * acc
        for axis in range(3):
            m[k, axis] = vk[axis]
    u[i] = m


@wp.func
def pair_flux(
    i: int,
    j: int,
    x: wp.array(dtype=wp.vec3),
    a: wp.array(dtype=wp.vec3),
    u: wp.array(dtype=wp.mat33),
    v: wp.array(dtype=wp.vec3),
    g: wp.vec3,
    dt: float,
    h: float,
    diffusion: float,
    nf: int,
):
    flux = wp.vec3(0.0)
    r = x[i] - x[j]
    for k in range(3):
        flux[k] = (
            dt
            * diffusion
            * (a[i][k] - a[j][k])
            * wp.dot(r, g)
            / (wp.dot(r, r) + 0.01 * h * h)
        )
        if i < nf and j < nf:
            drift = a[i][k] * (row(u[i], k) - v[i]) + a[j][k] * (row(u[j], k) - v[j])
            flux[k] -= dt * wp.dot(drift, g)
    # Correct roundoff in the most populated phase, leaving absent phases empty.
    complement = int(0)
    for k in range(3):
        if a[i][k] + a[j][k] < 1.0e-10:
            flux[k] = 0.0
        if a[i][k] + a[j][k] > a[i][complement] + a[j][complement]:
            complement = k
    flux[complement] -= flux[0] + flux[1] + flux[2]
    return flux


@wp.kernel(enable_backward=False)
def outgoing(
    x: wp.array(dtype=wp.vec3),
    a: wp.array(dtype=wp.vec3),
    u: wp.array(dtype=wp.mat33),
    v: wp.array(dtype=wp.vec3),
    ids: wp.array2d(dtype=int),
    grad: wp.array2d(dtype=wp.vec3),
    count: wp.array(dtype=int),
    limit: wp.array(dtype=wp.vec3),
    dt: float,
    h: float,
    diffusion: float,
    nf: int,
    na: int,
):
    i = wp.tid()
    loss = wp.vec3(0.0)
    for t in range(count[i]):
        j = ids[i, t]
        if j < na:
            flux = pair_flux(i, j, x, a, u, v, grad[i, t], dt, h, diffusion, nf)
            for k in range(3):
                loss[k] += wp.max(-flux[k], 0.0)
    l = wp.vec3(1.0)
    for k in range(3):
        if loss[k] > 0.0:
            l[k] = wp.min(1.0, 0.99999 * a[i][k] / loss[k])
    limit[i] = l


@wp.kernel(enable_backward=False)
def transport(
    x: wp.array(dtype=wp.vec3),
    a: wp.array(dtype=wp.vec3),
    u: wp.array(dtype=wp.mat33),
    v: wp.array(dtype=wp.vec3),
    ids: wp.array2d(dtype=int),
    grad: wp.array2d(dtype=wp.vec3),
    count: wp.array(dtype=int),
    limit: wp.array(dtype=wp.vec3),
    anew: wp.array(dtype=wp.vec3),
    unew: wp.array(dtype=wp.mat33),
    dt: float,
    h: float,
    diffusion: float,
    nf: int,
    na: int,
):
    i = wp.tid()
    ai = wp.vec3d(wp.float64(a[i][0]), wp.float64(a[i][1]), wp.float64(a[i][2]))
    mom = wp.mat33d(wp.float64(0.0))
    for k in range(3):
        for d in range(3):
            mom[k, d] = ai[k] * wp.float64(u[i][k, d])
    for t in range(count[i]):
        j = ids[i, t]
        if j < na:
            flux = pair_flux(i, j, x, a, u, v, grad[i, t], dt, h, diffusion, nf)
            scale = float(1.0)
            for k in range(3):
                if flux[k] < 0.0:
                    scale = wp.min(scale, limit[i][k])
                elif flux[k] > 0.0:
                    scale = wp.min(scale, limit[j][k])
            flux *= scale
            ai += wp.vec3d(
                wp.float64(flux[0]), wp.float64(flux[1]), wp.float64(flux[2])
            )
            for k in range(3):
                donor = i
                if flux[k] > 0.0:
                    donor = j
                for d in range(3):
                    mom[k, d] += wp.float64(flux[k]) * wp.float64(u[donor][k, d])
    anew[i] = wp.vec3(wp.float32(ai[0]), wp.float32(ai[1]), wp.float32(ai[2]))
    result = wp.mat33(0.0)
    for k in range(3):
        for d in range(3):
            if ai[k] > 1.0e-9:
                result[k, d] = wp.float32(mom[k, d] / ai[k])
            else:
                result[k, d] = v[i][d]
    unew[i] = result


@wp.func
def contact_velocity(m: wp.mat33, normal: wp.vec3, wall: wp.vec3):
    result = m
    for k in range(3):
        vk = row(m, k)
        vk -= wp.min(wp.dot(vk - wall, normal), 0.0) * normal
        for d in range(3):
            result[k, d] = vk[d]
    return result


@wp.kernel(enable_backward=False)
def advance(
    x: wp.array(dtype=wp.vec3),
    u: wp.array(dtype=wp.mat33),
    v: wp.array(dtype=wp.vec3),
    dt: float,
    lo: wp.vec3,
    hi: wp.vec3,
    radius: float,
    cylindrical: int,
    bag: wp.vec3,
    baghalf: wp.vec3,
    bagv: wp.vec3,
    rod: wp.vec3,
    rodv: wp.vec3,
    rodr: float,
    rodhalf: float,
    hasbag: int,
    contacts: wp.array(dtype=float),
):
    i = wp.tid()
    p = x[i] + dt * v[i]
    m = u[i]
    margin = radius * 0.45
    for d in range(3):
        if d == 2 or cylindrical == 0:
            n = wp.vec3(0.0)
            if p[d] < lo[d] + margin:
                wp.atomic_max(contacts, 0, lo[d] + margin - p[d])
                p[d] = lo[d] + margin
                n[d] = 1.0
            elif d < 2 and p[d] > hi[d] - margin:
                wp.atomic_max(contacts, 0, p[d] - (hi[d] - margin))
                p[d] = hi[d] - margin
                n[d] = -1.0
            m = contact_velocity(m, n, wp.vec3(0.0))
    if cylindrical == 1:
        rr = wp.sqrt(p[0] * p[0] + p[1] * p[1])
        if rr > hi[0] - margin:
            n = -wp.vec3(p[0], p[1], 0.0) / rr
            wp.atomic_max(contacts, 0, rr - (hi[0] - margin))
            p += (rr - (hi[0] - margin)) * n
            m = contact_velocity(m, n, wp.vec3(0.0))
    if hasbag == 1:
        q = p - bag
        gaps = (
            wp.vec3(wp.abs(q[0]), wp.abs(q[1]), wp.abs(q[2]))
            - baghalf
            - wp.vec3(margin)
        )
        if wp.max(gaps[0], wp.max(gaps[1], gaps[2])) < 0.0:
            axis = int(0)
            if gaps[1] > gaps[axis]:
                axis = 1
            if gaps[2] > gaps[axis]:
                axis = 2
            n = wp.vec3(0.0)
            n[axis] = 1.0
            if q[axis] < 0.0:
                n[axis] = -1.0
            p -= gaps[axis] * n
            m = contact_velocity(m, n, bagv)
            wp.atomic_max(contacts, 0, -gaps[axis])
    if rodr > 0.0 and wp.abs(p[2] - rod[2]) < rodhalf + margin:
        q = wp.vec3(p[0] - rod[0], p[1] - rod[1], 0.0)
        rr = wp.length(q)
        if rr < rodr + margin:
            depth = rodr + margin - rr
            n = wp.vec3(1.0, 0.0, 0.0)
            if rr > 1.0e-7:
                n = q / rr
            capdepth = rodhalf + margin - wp.abs(p[2] - rod[2])
            if capdepth < depth:
                n = wp.vec3(0.0, 0.0, wp.sign(p[2] - rod[2]))
                depth = capdepth
            p += depth * n
            m = contact_velocity(m, n, rodv)
            wp.atomic_max(contacts, 0, depth)
    x[i] = p
    u[i] = m


@wp.kernel(enable_backward=False)
def max_speed(u: wp.array(dtype=wp.mat33), speed: wp.array(dtype=float)):
    i = wp.tid()
    for k in range(3):
        wp.atomic_max(speed, 0, wp.length(row(u[i], k)))


class Mixture:
    def __init__(
        self,
        positions,
        fractions,
        velocities,
        kinds,
        local_positions,
        config,
        device="cuda:0",
    ):
        wp.init()
        self.cfg = config
        self.device = device
        self.dx = float(config["dx"])
        self.h = 2.3 * self.dx
        self.vol = self.dx**3
        self.nf = int(np.count_nonzero(kinds == 0))
        self.na = int(np.count_nonzero(kinds <= 2) - np.count_nonzero(kinds == 1))
        self.n = len(positions)
        self.cd = float(config.get("drag", 0.5))
        self.diffusion = float(config.get("diffusion", 0.0))
        self.density = wp.vec3(*config["densities"])
        self.nu = float(config.get("viscosity", 0.0001))
        self.x = wp.array(positions, dtype=wp.vec3, device=device)
        self.base = wp.array(local_positions, dtype=wp.vec3, device=device)
        self.a = wp.array(fractions, dtype=wp.vec3, device=device)
        self.u = wp.array(
            np.repeat(velocities[:, None, :], 3, axis=1), dtype=wp.mat33, device=device
        )
        self.v = wp.array(velocities, dtype=wp.vec3, device=device)
        self.kind = wp.array(kinds, dtype=int, device=device)
        self.rho = wp.zeros(self.n, dtype=float, device=device)
        self.mob = wp.zeros_like(self.rho)
        self.cap = 192
        self.ids = wp.zeros((self.na, self.cap), dtype=int, device=device)
        self.grad = wp.zeros((self.na, self.cap), dtype=wp.vec3, device=device)
        self.count = wp.zeros(self.na, dtype=int, device=device)
        self.psi = wp.zeros(self.na, dtype=float, device=device)
        self.diag = wp.zeros(self.nf, dtype=float, device=device)
        self.p = wp.zeros_like(self.diag)
        self.error = wp.zeros(2, dtype=float, device=device)
        self.overflow = wp.zeros(1, dtype=int, device=device)
        self.limit = wp.zeros(self.na, dtype=wp.vec3, device=device)
        self.anew = wp.clone(self.a)
        self.unew = wp.clone(self.u)
        self.speed = wp.zeros(1, dtype=float, device=device)
        self.contacts = wp.zeros_like(self.speed)
        self.grid = wp.HashGrid(128, 128, 128, device=device)
        self.last_projection = {}
        self.steps = 0
        self.remix()

    def launch(self, kernel, n, inputs):
        wp.launch(kernel, n, inputs=inputs, device=self.device)

    def remix(self):
        self.launch(
            mixture,
            self.n,
            [
                self.u,
                self.a,
                self.v,
                self.rho,
                self.mob,
                self.density,
                self.cd,
                self.nf,
            ],
        )

    def set_kinematics(self, state):
        self.motion = state
        self.launch(
            kinematics,
            self.n,
            [
                self.x,
                self.base,
                self.kind,
                self.v,
                self.u,
                *[
                    wp.vec3(*state[k])
                    for k in ["bag", "bag_velocity", "rod", "rod_velocity"]
                ],
            ],
        )

    def neighbors(self):
        self.grid.build(self.x, self.h)
        self.overflow.zero_()
        self.launch(
            neighbor_cache,
            self.na,
            [
                self.x,
                self.grid.id,
                self.h,
                self.vol,
                self.ids,
                self.grad,
                self.count,
                self.psi,
                self.overflow,
            ],
        )
        maximum = int(self.overflow.numpy()[0])
        if maximum > self.cap:
            raise RuntimeError(f"Neighbor capacity exceeded: {maximum} > {self.cap}")
        self.launch(
            diagonal, self.nf, [self.ids, self.grad, self.count, self.mob, self.diag]
        )

    def project(self, dt, stage):
        maximum = int(self.cfg.get("pressure_iterations", 48))
        tol = float(self.cfg.get("pressure_tolerance", 0.0002))
        for iteration in range(maximum):
            self.error.zero_()
            self.launch(
                pressure_residual,
                self.nf,
                [
                    self.v,
                    self.ids,
                    self.grad,
                    self.count,
                    self.psi,
                    self.diag,
                    self.p,
                    self.error,
                    dt,
                    stage,
                    0.5,
                ],
            )
            if iteration % 4 == 3 or iteration == maximum - 1:
                error = self.error.numpy()
                if float(error[0]) < tol and float(error[1]) < 0.025:
                    break
            self.launch(
                pressure_update,
                self.nf,
                [
                    self.u,
                    self.rho,
                    self.ids,
                    self.grad,
                    self.count,
                    self.p,
                    self.density,
                    self.cd,
                    self.nf,
                ],
            )
            self.remix()
        self.last_projection[str(stage)] = {
            "iterations": iteration + 1,
            "mean": float(error[0]),
            "maximum": float(error[1]),
        }

    def transfer(self, dt):
        self.launch(
            outgoing,
            self.na,
            [
                self.x,
                self.a,
                self.u,
                self.v,
                self.ids,
                self.grad,
                self.count,
                self.limit,
                dt,
                self.h,
                self.diffusion,
                self.nf,
                self.na,
            ],
        )
        self.launch(
            transport,
            self.na,
            [
                self.x,
                self.a,
                self.u,
                self.v,
                self.ids,
                self.grad,
                self.count,
                self.limit,
                self.anew,
                self.unew,
                dt,
                self.h,
                self.diffusion,
                self.nf,
                self.na,
            ],
        )
        self.a, self.anew = self.anew, self.a
        self.u, self.unew = self.unew, self.u
        self.remix()

    def timestep(self, remaining):
        self.speed.zero_()
        self.launch(max_speed, self.na, [self.u, self.speed])
        vmax = float(self.speed.numpy()[0])
        dt = min(
            float(self.cfg.get("dt", 0.002)), 0.22 * self.dx / max(vmax, 0.1), remaining
        )
        if self.nu > 0:
            dt = min(dt, 0.08 * self.dx**2 / self.nu)
        if not np.isfinite(vmax) or dt < 1.0e-6:
            raise RuntimeError(f"Invalid CFL state: speed={vmax}, dt={dt}")
        return dt

    def step(self, dt, motion):
        self.remix()
        self.set_kinematics(motion)
        self.neighbors()
        self.project(dt, 0)
        self.launch(
            advection_forces,
            self.nf,
            [
                self.x,
                self.u,
                self.v,
                self.ids,
                self.grad,
                self.count,
                dt,
                self.h,
                self.cd,
                self.nu,
                wp.vec3(*self.cfg.get("gravity", [0, 0, -9.81])),
            ],
        )
        self.remix()
        self.project(dt, 1)
        # Fluxes use the same positions and SPH gradients as the pressure solve.
        self.transfer(dt)
        c = self.cfg
        m = motion
        self.launch(
            advance,
            self.nf,
            [
                self.x,
                self.u,
                self.v,
                dt,
                wp.vec3(*c["bounds"][0]),
                wp.vec3(*c["bounds"][1]),
                self.dx,
                int(c.get("cylindrical", False)),
                wp.vec3(*m["bag"]),
                wp.vec3(*c.get("bag_half", [0, 0, 0])),
                wp.vec3(*m["bag_velocity"]),
                wp.vec3(*m["rod"]),
                wp.vec3(*m["rod_velocity"]),
                float(c.get("rod_radius", 0)),
                float(c.get("rod_half", 0)),
                int(c.get("tea_bag", False)),
                self.contacts,
            ],
        )
        self.remix()
        self.steps += 1

    def snapshot(self):
        return dict(
            positions=self.x.numpy()[: self.na],
            fractions=self.a.numpy()[: self.na],
            phase_velocities=self.u.numpy()[: self.na],
            velocities=self.v.numpy()[: self.na],
        )

    def diagnostics(self):
        s = self.snapshot()
        a = s["fractions"].astype(np.float64)
        u = s["phase_velocities"].astype(np.float64)
        masses = a * np.asarray(self.cfg["densities"]) * self.vol
        compression = self.psi.numpy()[: self.nf]
        return dict(
            steps=self.steps,
            phase_mass=masses.sum(0).tolist(),
            fluid_phase_mass=masses[: self.nf].sum(0).tolist(),
            momentum=(masses[..., None] * u).sum((0, 1)).tolist(),
            phase_min=float(a.min()),
            partition_error=float(abs(a.sum(1) - 1).max()),
            max_speed=float(np.linalg.norm(u, axis=-1).max()),
            compression_mean=float(np.maximum(compression - 1, 0).mean()),
            compression_max=float(compression.max() - 1),
            contact_correction=float(self.contacts.numpy()[0]),
            projection=self.last_projection,
        )
