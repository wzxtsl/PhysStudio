"""NumPy geometry helpers shared with the Blender renderer."""
import numpy as np


def star_vertices(radius):
    angle=np.arange(10)*np.pi/5+np.pi/2
    r=np.where(np.arange(10)%2==0,radius,radius*.52)
    return np.c_[r*np.cos(angle),r*np.sin(angle)].astype(np.float32)


def distance_and_reflection(x,radius):
    p=x[:,:2];vertices=star_vertices(radius);inside=np.zeros(len(p),bool)
    dist=np.full(len(p),np.inf);nearest=np.zeros_like(p)
    for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
        edge=b-a;t=np.clip(((p-a)*edge).sum(1)/np.dot(edge,edge),0,1)
        q=a+t[:,None]*edge;d=np.linalg.norm(p-q,axis=1);update=d<dist
        dist[update]=d[update];nearest[update]=q[update]
        if abs(b[1]-a[1])>1.e-12:
            hit=a[0]+(p[:,1]-a[1])*(b[0]-a[0])/(b[1]-a[1])
            inside^=((a[1]>p[:,1])!=(b[1]>p[:,1]))&(p[:,0]<hit)
    reflected=x.copy();reflected[:,:2]=2*nearest-p
    return np.where(inside,-dist,dist),reflected
