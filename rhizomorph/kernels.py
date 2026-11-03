"""Warp finite-volume transport, root uptake, sensing, and meristem dynamics."""
import warp as wp


@wp.func
def index3(i: int, j: int, k: int, nx: int, ny: int):
    return (k * ny + j) * nx + i


@wp.func
def sample(
    a: wp.array(dtype=float),
    p: wp.vec3,
    lo: wp.vec3,
    h: wp.vec3,
    nx: int,
    ny: int,
    nz: int,
):
    q = wp.cw_div(p - lo, h) - wp.vec3(0.5, 0.5, 0.5)
    x = wp.clamp(q[0], 0.0, float(nx - 1) - 0.0001)
    y = wp.clamp(q[1], 0.0, float(ny - 1) - 0.0001)
    z = wp.clamp(q[2], 0.0, float(nz - 1) - 0.0001)
    i = int(wp.floor(x))
    j = int(wp.floor(y))
    k = int(wp.floor(z))
    f = wp.vec3(x - float(i), y - float(j), z - float(k))
    v = float(0.0)
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                w = (
                    wp.abs(float(1 - dx) - f[0])
                    * wp.abs(float(1 - dy) - f[1])
                    * wp.abs(float(1 - dz) - f[2])
                )
                v += w * a[index3(i + dx, j + dy, k + dz, nx, ny)]
    return v


@wp.func
def gradient(
    a: wp.array(dtype=float),
    p: wp.vec3,
    lo: wp.vec3,
    h: wp.vec3,
    nx: int,
    ny: int,
    nz: int,
):
    g = wp.vec3(0.0)
    for axis in range(3):
        d = wp.vec3(0.0)
        d[axis] = h[axis]
        g[axis] = (
            sample(a, p + d, lo, h, nx, ny, nz) - sample(a, p - d, lo, h, nx, ny, nz)
        ) / (2.0 * h[axis])
    return g


@wp.kernel
def face_flux(
    w: wp.array(dtype=float),
    n: wp.array(dtype=float),
    c: wp.array(dtype=float),
    perm: wp.array(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    h: wp.vec3,
    dt: float,
    diffusion: float,
    gravity: float,
    fw: wp.array(dtype=wp.vec3),
    fn: wp.array(dtype=wp.vec3),
):
    a = wp.tid()
    i = a % nx
    j = (a // nx) % ny
    k = a // (nx * ny)
    wf = wp.vec3(0.0)
    nf = wp.vec3(0.0)
    for axis in range(3):
        b = int(a + 1)
        valid = bool(i < nx - 1)
        if axis == 1:
            b = a + nx
            valid = j < ny - 1
        if axis == 2:
            b = a + nx * ny
            valid = k < nz - 1
        if valid:
            permeability = 2.0 * perm[a] * perm[b] / wp.max(perm[a] + perm[b], 1.0e-12)
            q = -diffusion * (w[b] / c[b] - w[a] / c[a]) / h[axis]
            if axis == 2:
                q -= gravity * w[b]
            transfer = dt * permeability * q / h[axis]
            # Six-face donor limiter guarantees positivity while sharing each face flux.
            transfer = wp.clamp(transfer, -w[b] * 0.16, w[a] * 0.16)
            donor = int(a)
            if transfer < 0.0:
                donor = b
            nutrient = transfer * n[donor]
            nutrient = wp.clamp(nutrient, -n[b] * 0.16, n[a] * 0.16)
            wf[axis] = transfer
            nf[axis] = nutrient
    fw[a] = wf
    fn[a] = nf


@wp.kernel
def divergence(
    w: wp.array(dtype=float),
    n: wp.array(dtype=float),
    fw: wp.array(dtype=wp.vec3),
    fn: wp.array(dtype=wp.vec3),
    nx: int,
    ny: int,
    nz: int,
    wo: wp.array(dtype=float),
    no: wp.array(dtype=float),
):
    a = wp.tid()
    i = a % nx
    j = (a // nx) % ny
    k = a // (nx * ny)
    dw = -fw[a][0] - fw[a][1] - fw[a][2]
    dn = -fn[a][0] - fn[a][1] - fn[a][2]
    if i > 0:
        dw += fw[a - 1][0]
        dn += fn[a - 1][0]
    if j > 0:
        dw += fw[a - nx][1]
        dn += fn[a - nx][1]
    if k > 0:
        dw += fw[a - nx * ny][2]
        dn += fn[a - nx * ny][2]
    wo[a] = w[a] + dw
    no[a] = n[a] + dn


@wp.kernel
def boundary(
    w: wp.array(dtype=float),
    n: wp.array(dtype=float),
    perm: wp.array(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    rain: float,
    drain: float,
    ledger: wp.array(dtype=wp.float64),
):
    a = wp.tid()
    k = a // (nx * ny)
    if k == nz - 1 and perm[a] > 0.0:
        w[a] += rain
        wp.atomic_add(ledger, 0, wp.float64(rain))
    if k < 3:
        fraction = drain * float(3 - k) / 3.0
        loss = w[a] * fraction
        nl = n[a] * fraction
        w[a] -= loss
        n[a] -= nl
        wp.atomic_add(ledger, 1, wp.float64(loss))
        wp.atomic_add(ledger, 3, wp.float64(nl))


@wp.kernel
def uptake_request(
    p: wp.array(dtype=wp.vec3),
    parent: wp.array(dtype=int),
    kind: wp.array(dtype=int),
    order: wp.array(dtype=int),
    w: wp.array(dtype=float),
    c: wp.array(dtype=float),
    lo: wp.vec3,
    h: wp.vec3,
    nx: int,
    ny: int,
    nz: int,
    rate: float,
    demand: wp.array(dtype=float),
    cell: wp.array(dtype=int),
    request: wp.array(dtype=float),
):
    a = wp.tid()
    request[a] = 0.0
    cell[a] = 0
    if kind[a] == 1 and parent[a] >= 0:
        q = wp.cw_div(p[a] - lo, h)
        x = wp.clamp(int(q[0]), 0, nx - 1)
        y = wp.clamp(int(q[1]), 0, ny - 1)
        z = wp.clamp(int(q[2]), 0, nz - 1)
        b = index3(x, y, z, nx, ny)
        cell[a] = b
        absorption = (
            rate
            * (0.25 + float(order[a]) * 0.6)
            * wp.length(p[a] - p[parent[a]])
            / 0.05
        )
        absorption *= wp.clamp(w[b] / c[b] / 0.65, 0.0, 1.0)
        request[a] = absorption
        wp.atomic_add(demand, b, absorption)


@wp.kernel
def uptake_actual(
    request: wp.array(dtype=float),
    cell: wp.array(dtype=int),
    demand: wp.array(dtype=float),
    w: wp.array(dtype=float),
    actual: wp.array(dtype=float),
):
    a = wp.tid()
    b = cell[a]
    actual[a] = request[a] * wp.min(1.0, w[b] / wp.max(demand[b], 1.0e-12))


@wp.kernel
def uptake_apply(
    w: wp.array(dtype=float),
    demand: wp.array(dtype=float),
    ledger: wp.array(dtype=wp.float64),
):
    a = wp.tid()
    v = wp.min(w[a], demand[a])
    w[a] -= v
    wp.atomic_add(ledger, 2, wp.float64(v))


@wp.kernel
def sense(
    p: wp.array(dtype=wp.vec3),
    kind: wp.array(dtype=int),
    order: wp.array(dtype=int),
    age: wp.array(dtype=float),
    w: wp.array(dtype=float),
    nu: wp.array(dtype=float),
    resistance: wp.array(dtype=float),
    lo: wp.vec3,
    h: wp.vec3,
    nx: int,
    ny: int,
    nz: int,
    light_scale: float,
    light: wp.array(dtype=float),
    soil: wp.array(dtype=wp.vec3),
    optimal: wp.array(dtype=wp.vec3),
):
    a = wp.tid()
    if kind[a] == 1:
        soil[a] = wp.vec3(
            sample(w, p[a], lo, h, nx, ny, nz),
            sample(nu, p[a], lo, h, nx, ny, nz),
            sample(resistance, p[a], lo, h, nx, ny, nz),
        )
        light[a] = 0.0
        optimal[a] = wp.vec3(0.0, 0.0, -1.0)
    else:
        total = float(0.0)
        incoming = wp.vec3(0.0)
        for ray in range(5):
            angle = float(ray) * 2.3999632
            direction = wp.normalize(
                wp.vec3(0.45 * wp.cos(angle), 0.45 * wp.sin(angle), 1.0)
            )
            optical = float(0.0)
            for b in range(p.shape[0]):
                if kind[b] == 0 and order[b] > 0 and age[b] > 1.0:
                    delta = p[b] - p[a]
                    z = wp.dot(delta, direction)
                    rr = wp.length_sq(delta) - z * z
                    if z > 0.08 and rr < 0.10:
                        optical += 0.11 * wp.exp(-rr / 0.025)
            transmission = wp.exp(-optical)
            total += transmission
            incoming += direction * transmission
        light[a] = light_scale * total / 5.0
        optimal[a] = wp.normalize(incoming)
        soil[a] = wp.vec3(0.0, 0.0, 1.0)


@wp.kernel
def accumulate(
    tip: wp.array(dtype=int),
    parent: wp.array(dtype=int),
    kind: wp.array(dtype=int),
    order: wp.array(dtype=int),
    soil: wp.array(dtype=wp.vec3),
    light: wp.array(dtype=float),
    weight: wp.array(dtype=float),
    decay: float,
):
    a = wp.tid()
    if tip[a] == 1:
        value = wp.max(0.02, light[a])
        if kind[a] == 1:
            value = wp.max(0.02, soil[a][1]) / wp.max(1.0, soil[a][2])
        value *= wp.pow(0.82, float(order[a]))
        node = int(a)
        while node >= 0:
            wp.atomic_add(weight, node, value)
            value *= decay
            node = parent[node]


@wp.kernel
def elongate(
    p: wp.array(dtype=wp.vec3),
    heading: wp.array(dtype=wp.vec3),
    kind: wp.array(dtype=int),
    order: wp.array(dtype=int),
    tip: wp.array(dtype=int),
    parent: wp.array(dtype=int),
    weight: wp.array(dtype=float),
    siblings: wp.array(dtype=float),
    local_fraction: wp.array(dtype=float),
    optimal: wp.array(dtype=wp.vec3),
    nu: wp.array(dtype=float),
    resistance: wp.array(dtype=float),
    sdf: wp.array(dtype=float),
    lo: wp.vec3,
    h: wp.vec3,
    nx: int,
    ny: int,
    nz: int,
    seed: int,
    step: int,
    style: int,
    shoot_length: float,
    root_length: float,
    shoot_budget: float,
    root_budget: float,
    out: wp.array(dtype=wp.vec3),
    direction: wp.array(dtype=wp.vec3),
    allocation: wp.array(dtype=float),
):
    a = wp.tid()
    out[a] = p[a]
    direction[a] = heading[a]
    allocation[a] = 0.0
    if tip[a] == 1:
        budget = shoot_budget
        if kind[a] == 1:
            budget = root_budget
        node = int(a)
        while parent[node] >= 0:
            budget *= weight[node] / wp.max(siblings[node], 1.0e-12)
            node = parent[node]
        budget *= local_fraction[a]
        allocation[a] = budget
        rate = wp.min(1.15, budget)
        state = wp.rand_init(seed + a * 971, step)
        noise = wp.vec3(wp.randn(state), wp.randn(state), wp.randn(state))
        d = heading[a]
        length = shoot_length
        if kind[a] == 0:
            gravity = -0.012
            photo = 0.018
            if order[a] == 0:
                gravity = 0.11
                photo = 0.045
            d = wp.normalize(
                d + wp.vec3(0.0, 0.0, gravity) + photo * optimal[a] + 0.055 * noise
            )
            length *= wp.pow(0.90, float(order[a]))
            if order[a] == 0:
                length *= 0.55
            if order[a] == 1:
                length *= 1.45
        else:
            length = root_length * wp.pow(0.86, float(order[a]))
            if order[a] == 1:
                length *= 2.1
            g = gradient(nu, p[a], lo, h, nx, ny, nz)
            g = g / (1.0 + wp.length(g))
            obstacle = gradient(sdf, p[a], lo, h, nx, ny, nz)
            dist = sample(sdf, p[a], lo, h, nx, ny, nz)
            repel = wp.vec3(0.0)
            if dist < 0.18:
                repel = wp.normalize(obstacle) * (0.18 - dist) * 2.5
            down = 0.014
            if order[a] == 1:
                down = 0.006
            if style == 0 and order[a] == 0:
                down = 0.2
            if style == 2 and order[a] < 2:
                down = 0.002
            if style == 2 and order[a] == 2:
                down = 0.10
            d = wp.normalize(
                d + 0.17 * g + repel + wp.vec3(0.0, 0.0, -down) + 0.09 * noise
            )
            if dist < 0.22 and wp.dot(d, obstacle) < 0.0:
                normal = wp.normalize(obstacle)
                tangent = d - normal * wp.dot(d, normal) + 0.12 * noise
                if wp.length(tangent) < 0.05:
                    tangent = wp.cross(normal, wp.vec3(1.0, 0.3, 0.2))
                d = wp.normalize(tangent + 0.10 * normal)
            length /= wp.sqrt(wp.max(1.0, sample(resistance, p[a], lo, h, nx, ny, nz)))
            if p[a][2] > -0.13 and d[2] > -0.1:
                d = wp.normalize(wp.vec3(d[0], d[1], -0.22))
            # Select an obstacle-free swept segment, not just a valid endpoint.
            best = wp.vec3(0.0)
            best_score = float(-1.0e10)
            for trial in range(16):
                candidate = d
                if trial > 0:
                    candidate = wp.normalize(
                        d
                        + 0.75
                        * wp.vec3(wp.randn(state), wp.randn(state), wp.randn(state))
                    )
                endpoint = p[a] + candidate * length * rate
                valid = bool(True)
                for q in range(1, 5):
                    pos = p[a] + candidate * length * rate * float(q) / 4.0
                    if sample(sdf, pos, lo, h, nx, ny, nz) < 0.065:
                        valid = False
                if (
                    endpoint[2] > -0.025
                    or endpoint[2] < lo[2] + 0.14
                    or wp.abs(endpoint[0]) > 3.38
                    or wp.abs(endpoint[1]) > 3.38
                ):
                    valid = False
                score = wp.dot(candidate, d)
                if valid and score > best_score:
                    best_score = score
                    best = candidate
            if best_score < -1.0e9:
                rate = 0.0
            else:
                d = best
        out[a] = p[a] + d * length * rate
        direction[a] = d
