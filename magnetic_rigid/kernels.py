"""Magnetic incremental potentials and derivatives, independently written in Warp.

Reference: Westhofen et al., CGF 2024, DOI 10.1111/cgf.15185, Eqs. 3--26.
No STARK code is imported or translated. Distances use analytic sphere unions.
"""
import warp as wp

D = wp.float64
V = wp.vec3d
Q = wp.quatd
M = wp.mat33d
U12 = wp.types.vector(length=12, dtype=D)
MU_OVER_4PI = wp.constant(wp.float64(1.e-7))
FOUR_PI = wp.constant(wp.float64(12.566370614359172))


@wp.struct
class Term:
    kind: int
    a: int
    b: int
    ra: V
    rb: V
    ma: V
    mb: V
    normal: V
    bias: V
    radius: D
    stiffness: D
    thickness: D
    friction: D
    epsilon: D


@wp.func
def orientation(q: Q, w: V, dt: D):
    dq = Q(w[0] * dt * D(0.5), w[1] * dt * D(0.5), w[2] * dt * D(0.5), D(1.0))
    return wp.normalize(dq * q)


@wp.func
def pose(u: U12, off: int, idx: int, x: wp.array(dtype=V), q: wp.array(dtype=Q),
         sv: wp.array(dtype=V), sw: wp.array(dtype=V), dt: D):
    v = V(u[off], u[off+1], u[off+2])
    w = V(u[off+3], u[off+4], u[off+5])
    return x[idx] + dt * wp.cw_mul(v, sv[idx]), orientation(q[idx], wp.cw_mul(w, sw[idx]), dt)


@wp.func
def pack(u: wp.array(dtype=D), a: int, b: int):
    y = U12()
    for k in range(6):
        y[k] = u[6*a+k]
        y[6+k] = u[6*b+k]
    return y


@wp.func
def value(t: Term, u: U12, x: wp.array(dtype=V), q: wp.array(dtype=Q),
          sv: wp.array(dtype=V), sw: wp.array(dtype=V), dt: D):
    xa, qa = pose(u, 0, t.a, x, q, sv, sw, dt)
    xb, qb = pose(u, 6, t.b, x, q, sv, sw, dt)
    pa = xa + wp.quat_rotate(qa, t.ra)
    pb = xb + wp.quat_rotate(qb, t.rb)
    result = D(0.0)
    if t.kind == 0:
        # One unordered pair equals the two directed -1/2 m_i . B_j terms.
        d = pa - pb
        rr = wp.dot(d, d)
        invr = D(1.0) / wp.sqrt(rr)
        ma = wp.quat_rotate(qa, t.ma)
        mb = wp.quat_rotate(qb, t.mb)
        result = MU_OVER_4PI * invr * invr * invr * (wp.dot(ma, mb) - D(3.0) * wp.dot(ma, d) * wp.dot(mb, d) / rr)
    elif t.kind <= 3:
        distance = D(0.0)
        if t.kind == 1:
            distance = wp.length(pa - pb) - t.radius
        elif t.kind == 2:
            distance = wp.dot(pa, t.normal) - t.radius
        else:
            distance = t.radius - wp.length(pa - t.rb)
        if distance < t.thickness:
            gap = t.thickness - distance
            result = t.stiffness * gap * gap * gap
    elif t.kind == 4:
        c = pa - pb
        result = D(0.5) * t.stiffness * wp.dot(c, c)
    elif t.kind == 5:
        stretch = wp.length(pa - pb) - t.radius
        result = D(0.5) * t.stiffness * stretch * stretch
    elif t.kind == 6:
        # Lagged contact frame/load; both rigid contact connectors move.
        delta = pa - pb - t.bias
        tangent = delta - t.normal * wp.dot(t.normal, delta)
        length = wp.sqrt(wp.dot(tangent, tangent) + D(1.e-24))
        if length < t.epsilon:
            result = t.friction * (-length*length*length / (D(3.0)*t.epsilon*t.epsilon)
                                  + length*length/t.epsilon + t.epsilon/D(3.0))
        else:
            result = t.friction * length
    return result


@wp.kernel
def potentials(u: wp.array(dtype=D), terms: wp.array(dtype=Term),
               x: wp.array(dtype=V), q: wp.array(dtype=Q), sv: wp.array(dtype=V),
               sw: wp.array(dtype=V), dt: D, energy: wp.array(dtype=D)):
    i = wp.tid()
    t = terms[i]
    e = value(t, pack(u, t.a, t.b), x, q, sv, sw, dt)
    wp.atomic_add(energy, 0, e)


@wp.kernel
def inertia(u: wp.array(dtype=D), predicted: wp.array(dtype=D),
            J: wp.array(dtype=M), damping: wp.array(dtype=V), dt: D,
            far_gradient: wp.array(dtype=D), energy: wp.array(dtype=D)):
    b = wp.tid()
    v = V(u[6*b], u[6*b+1], u[6*b+2])
    w = V(u[6*b+3], u[6*b+4], u[6*b+5])
    dv = v - V(predicted[6*b], predicted[6*b+1], predicted[6*b+2])
    dw = w - V(predicted[6*b+3], predicted[6*b+4], predicted[6*b+5])
    e = D(0.5) * (wp.dot(dv, dv) + wp.dot(dw, J[b]*dw))
    e += D(0.5)*dt*(damping[b][0]*wp.dot(v,v) + damping[b][1]*wp.dot(w,J[b]*w))
    for k in range(6):
        e += far_gradient[6*b+k] * u[6*b+k]
    wp.atomic_add(energy, 0, e)


@wp.kernel(enable_backward=False)
def term_hessians(u: wp.array(dtype=D), terms: wp.array(dtype=Term),
                  x: wp.array(dtype=V), q: wp.array(dtype=Q), sv: wp.array(dtype=V),
                  sw: wp.array(dtype=V), dt: D, epsilon: D,
                  active: wp.array(dtype=int), hessian: wp.array2d(dtype=D)):
    t_idx, i, j = wp.tid()
    t = terms[t_idx]
    ai = t.a
    aj = t.a
    if i >= 6:
        ai = t.b
    if j >= 6:
        aj = t.b
    gi = 6*ai + i%6
    gj = 6*aj + j%6
    if active[gi] == 0 or active[gj] == 0 or j < i:
        return
    if t.kind >= 2 and t.kind <= 3 and (i >= 6 or j >= 6):
        return
    y = pack(u, t.a, t.b)
    hi = epsilon
    hj = epsilon
    second = D(0.0)
    if i == j:
        yp = y
        ym = y
        yp[i] += hi
        ym[i] -= hi
        second = (value(t,yp,x,q,sv,sw,dt) - D(2.0)*value(t,y,x,q,sv,sw,dt) + value(t,ym,x,q,sv,sw,dt))/(hi*hi)
    else:
        pp = y
        pm = y
        mp = y
        mm = y
        pp[i] += hi
        pp[j] += hj
        pm[i] += hi
        pm[j] -= hj
        mp[i] -= hi
        mp[j] += hj
        mm[i] -= hi
        mm[j] -= hj
        second = (value(t,pp,x,q,sv,sw,dt)-value(t,pm,x,q,sv,sw,dt)-value(t,mp,x,q,sv,sw,dt)+value(t,mm,x,q,sv,sw,dt))/(D(4.0)*hi*hj)
    wp.atomic_add(hessian, gi, gj, second)
    if gi != gj:
        wp.atomic_add(hessian, gj, gi, second)


@wp.kernel(enable_backward=False)
def induced_moments(owner: wp.array(dtype=int), local: wp.array(dtype=V),
                    x: wp.array(dtype=V), q: wp.array(dtype=Q),
                    old_m: wp.array(dtype=V), permanent: wp.array(dtype=V),
                    response: wp.array(dtype=D), count: int, new_m: wp.array(dtype=V)):
    i = wp.tid()
    if response[i] == D(0.0):
        new_m[i] = permanent[i]
    else:
        a = owner[i]
        pi = x[a] + wp.quat_rotate(q[a], local[i])
        field = V(D(0.0))
        for j in range(count):
            if owner[j] != a:
                b = owner[j]
                r = pi - (x[b] + wp.quat_rotate(q[b], local[j]))
                d = wp.length(r)
                n = r / d
                m = wp.quat_rotate(q[b], old_m[j])
                # H, not B: response = V chi/(1+chi/3), with SI units.
                field += (D(3.0)*n*wp.dot(n,m)-m) / (FOUR_PI*d*d*d)
        new_m[i] = wp.quat_rotate(wp.quat_inverse(q[a]), field * response[i])


@wp.kernel(enable_backward=False)
def integrate(u: wp.array(dtype=D), x: wp.array(dtype=V), q: wp.array(dtype=Q),
              sv: wp.array(dtype=V), sw: wp.array(dtype=V), dt: D,
              nx: wp.array(dtype=V), nq: wp.array(dtype=Q),
              velocity: wp.array(dtype=V), omega: wp.array(dtype=V)):
    i = wp.tid()
    v = wp.cw_mul(V(u[6*i],u[6*i+1],u[6*i+2]), sv[i])
    w = wp.cw_mul(V(u[6*i+3],u[6*i+4],u[6*i+5]), sw[i])
    nx[i] = x[i] + dt*v
    nq[i] = orientation(q[i],w,dt)
    velocity[i] = v
    omega[i] = w
