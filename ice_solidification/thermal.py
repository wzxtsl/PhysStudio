"""Backward-Euler SPH conduction with an exact latent-heat plateau."""
import math
import numpy as np
import warp as wp
from ice_melting import thermal as material

MAX_NEIGHBORS = 160


@wp.func
def box_distance(p: wp.vec3, center: wp.vec3, half: wp.vec3):
    q = wp.abs(p-center)-half
    return wp.length(wp.max(q, wp.vec3(0.)))+wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.)


@wp.func
def contact_distance(p: wp.vec3, mode: int, rod: wp.vec3, radius: float):
    distance = p[2]
    if mode == 1:
        distance = wp.length(wp.vec3(p[0]-rod[0], p[1]-rod[1], 0.))-radius
    elif mode == 2:
        distance = wp.min(distance, box_distance(p, wp.vec3(-.0575, 0., .013), wp.vec3(.0325, .032, .013)))
        distance = wp.min(distance, box_distance(p, wp.vec3(0., 0., .007), wp.vec3(.025, .032, .007)))
    return wp.max(distance, 0.)


@wp.kernel(enable_backward=False)
def setup(x: wp.array(dtype=wp.vec3), e: wp.array(dtype=float), grid: wp.uint64,
          h: float, norm: float, dx: float, mass: float, mode: int, rod: wp.vec3,
          rod_radius: float, cold_t: float, contact_h: float, air_t: float, air_h: float,
          ids: wp.array2d(dtype=int), weights: wp.array2d(dtype=float), count: wp.array(dtype=int),
          diagonal: wp.array(dtype=float), boundary_d: wp.array(dtype=float),
          boundary_b: wp.array(dtype=float), overflow: wp.array(dtype=int)):
    i = wp.tid(); p = x[i]; vi = material.volume(e[i], mass)
    fi = material.liquid(e[i]); ki = 2.2*(1.-fi)+.6*fi
    n = int(0); total = float(0.); occupancy = float(0.)
    for j in wp.hash_grid_query(grid, p, h):
        r = p-x[j]; r2 = wp.dot(r, r)
        if r2 < h*h:
            vj = material.volume(e[j], mass)
            q = 1.-r2/(h*h)
            occupancy += vj*q*q*q/(norm*dx*dx*dx)
            if j != i:
                fj = material.liquid(e[j]); kj = 2.2*(1.-fj)+.6*fj
                grad = -6.*q*q/(h*h*norm*dx*dx*dx)*r
                g = 4.*ki*kj/(ki+kj)*vi*vj*(-wp.dot(r, grad))/(r2+.01*h*h)
                if n < 160:
                    ids[i,n] = j; weights[i,n] = g; total += g; n += 1
                else:
                    wp.atomic_add(overflow, 0, 1)
    distance = contact_distance(p, mode, rod, rod_radius)
    contact = contact_h*dx*dx*wp.clamp((1.7*dx-distance)/(1.3*dx), 0., 1.)
    exposed = air_h*3.*dx*dx*wp.clamp((1.-occupancy)/.5, 0., 1.)
    count[i] = n; diagonal[i] = total
    boundary_d[i] = contact+exposed
    boundary_b[i] = contact*cold_t+exposed*air_t


@wp.kernel(enable_backward=False)
def jacobi(old: wp.array(dtype=float), current: wp.array(dtype=float), result: wp.array(dtype=float),
           ids: wp.array2d(dtype=int), weights: wp.array2d(dtype=float), count: wp.array(dtype=int),
           diagonal: wp.array(dtype=float), boundary_d: wp.array(dtype=float),
           boundary_b: wp.array(dtype=float), dt: float, mass: float):
    i = wp.tid(); m=wp.float64(mass);step=wp.float64(dt)
    rhs=m*wp.float64(old[i])+step*wp.float64(boundary_b[i])
    d=step*wp.float64(boundary_d[i])
    for k in range(count[i]):
        g=step*wp.float64(weights[i,k]);d+=g
        rhs+=g*wp.float64(material.temp(current[ids[i,k]]))
    value=rhs/m
    if rhs<wp.float64(0.):
        value=wp.float64(2100.)*rhs/(m*wp.float64(2100.)+d)
    elif rhs>m*wp.float64(333500.):
        value=wp.float64(333500.)+wp.float64(4180.)*(rhs-m*wp.float64(333500.))/(m*wp.float64(4180.)+d)
    result[i]=wp.float32(value)


@wp.kernel(enable_backward=False)
def residual(old: wp.array(dtype=float), current: wp.array(dtype=float),
             ids: wp.array2d(dtype=int), weights: wp.array2d(dtype=float), count: wp.array(dtype=int),
             boundary_d: wp.array(dtype=float), boundary_b: wp.array(dtype=float),
             dt: float, mass: float, totals: wp.array(dtype=wp.float64)):
    i = wp.tid();ti=wp.float64(material.temp(current[i]))
    boundary=wp.float64(boundary_b[i])-wp.float64(boundary_d[i])*ti;flux=boundary
    for k in range(count[i]):
        flux+=wp.float64(weights[i,k])*(wp.float64(material.temp(current[ids[i,k]]))-ti)
    r=wp.float64(mass)*(wp.float64(current[i])-wp.float64(old[i]))-wp.float64(dt)*flux
    wp.atomic_add(totals,0,wp.abs(r))
    wp.atomic_add(totals,1,wp.float64(dt)*boundary)


class ImplicitHeat:
    def __init__(self, system):
        self.system = system; self.device = system.device; n = system.n
        self.ids = wp.zeros((n, MAX_NEIGHBORS), dtype=int, device=self.device)
        self.weights = wp.zeros((n, MAX_NEIGHBORS), dtype=float, device=self.device)
        self.count = wp.zeros(n, dtype=int, device=self.device)
        self.diagonal = wp.zeros(n, dtype=float, device=self.device)
        self.bd = wp.zeros(n, dtype=float, device=self.device); self.bb = wp.zeros_like(self.bd)
        self.old = wp.zeros_like(self.bd); self.scratch = wp.zeros_like(self.bd)
        self.overflow = wp.zeros(1, dtype=int, device=self.device)
        self.totals = wp.zeros(2, dtype=wp.float64, device=self.device)
        self.external_energy = 0.; self.maximum_residual = 0.; self.maximum_iterations = 0
        self.last_iterations = 0

    def advance(self, dt, cold_t, depth=0):
        s = self.system; cfg = s.cfg
        s.grid.build(s.x, s.h); self.overflow.zero_(); wp.copy(self.old, s.e)
        s.launch(setup, s.n, [s.x, self.old, s.grid.id, s.h, s.norm, s.dx, s.mass,
            cfg['boundary_mode'], wp.vec3(*cfg.get('rod_center',[0.,0.,0.])), cfg.get('rod_radius',0.),
            cold_t, cfg['contact_h'], cfg.get('air_t',2.), cfg.get('air_h',0.),
            self.ids, self.weights, self.count, self.diagonal, self.bd, self.bb, self.overflow])
        if int(self.overflow.numpy()[0]):raise RuntimeError('Thermal neighbor capacity exceeded')
        tolerance = 3.e-7*s.mass*s.n*material.LATENT
        args = [self.ids, self.weights, self.count, self.diagonal, self.bd, self.bb, dt, s.mass]
        for iteration in range(1, 769):
            s.launch(jacobi, s.n, [self.old, s.e, self.scratch, *args])
            s.e, self.scratch = self.scratch, s.e
            if iteration % 16 == 0:
                self.totals.zero_()
                s.launch(residual,s.n,[self.old,s.e,self.ids,self.weights,self.count,self.bd,self.bb,dt,s.mass,self.totals])
                error, external = self.totals.numpy()
                if error <= tolerance:
                    self.external_energy += float(external)
                    self.maximum_residual = max(self.maximum_residual,float(error))
                    self.maximum_iterations = max(self.maximum_iterations, iteration)
                    self.last_iterations = iteration
                    return
        if depth >= 6:raise RuntimeError(f'Implicit heat solve failed: residual {error}, tolerance {tolerance}')
        wp.copy(s.e,self.old)
        self.advance(dt/2,cold_t,depth+1)
        self.advance(dt/2,cold_t,depth+1)
