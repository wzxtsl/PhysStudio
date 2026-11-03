"""Polygonal cold-mold geometry shared by simulation and reconstruction."""
from .geometry import star_vertices,distance_and_reflection
import warp as wp


@wp.func
def vertex(i:int,radius:float):
    a=float(i)*3.141592653589793/5.+3.141592653589793/2.
    r=radius
    if i%2==1:r*=.52
    return wp.vec2(r*wp.cos(a),r*wp.sin(a))


@wp.func
def star_distance(p:wp.vec2,radius:float):
    d=float(1.e6);inside=int(0)
    for i in range(10):
        a=vertex(i,radius);b=vertex((i+1)%10,radius);edge=b-a
        t=wp.clamp(wp.dot(p-a,edge)/wp.dot(edge,edge),0.,1.)
        d=wp.min(d,wp.length(p-a-t*edge))
        if (a[1]>p[1])!=(b[1]>p[1]):
            hit=a[0]+(p[1]-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if p[0]<hit:inside=1-inside
    if inside!=0:d=-d
    return d


@wp.kernel(enable_backward=False)
def project_star(x:wp.array(dtype=wp.vec3),radius:float,particle_radius:float):
    i=wp.tid();p=wp.vec2(x[i][0],x[i][1])
    if star_distance(p,radius)>-particle_radius:
        best=float(1.e6);nearest=wp.vec2(0.);normal=wp.vec2(0.)
        for k in range(10):
            a=vertex(k,radius);b=vertex((k+1)%10,radius);edge=b-a
            t=wp.clamp(wp.dot(p-a,edge)/wp.dot(edge,edge),0.,1.);q=a+t*edge
            d=wp.length(p-q)
            if d<best:
                best=d;nearest=q;normal=wp.normalize(wp.vec2(edge[1],-edge[0]))
        p=nearest-particle_radius*normal
        # Resolve the second face at sharp tips before freezing an anchor.
        for repeat in range(12):
            best=float(1.e6);qbest=wp.vec2(0.);nbest=wp.vec2(0.)
            for k in range(10):
                a=vertex(k,radius);b=vertex((k+1)%10,radius);edge=b-a
                t=wp.clamp(wp.dot(p-a,edge)/wp.dot(edge,edge),0.,1.);q=a+t*edge
                d=wp.length(p-q)
                if d<best:
                    best=d;qbest=q;nbest=wp.normalize(wp.vec2(edge[1],-edge[0]))
            if best<particle_radius or star_distance(p,radius)>0.:
                p=qbest-particle_radius*nbest
        x[i]=wp.vec3(p[0],p[1],x[i][2])
