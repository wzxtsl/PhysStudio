"""Continuum surface force with reflected support at a neutral-wetting plate."""
import warp as wp
from ice_melting import thermal
from ice_melting import solver as fluid


@wp.kernel(enable_backward=False)
def plate_density(x:wp.array(dtype=wp.vec3),e:wp.array(dtype=float),groups:wp.array(dtype=int),
                  grid:wp.uint64,h:float,norm:float,lam:wp.array(dtype=float),rho:wp.array(dtype=float)):
    i=wp.tid();p=x[i];d=float(0.);gsum=wp.vec3(0.);g2=float(0.)
    for j in wp.hash_grid_query(grid,p,h):
        volume=1.-.083*thermal.liquid(e[j]);r=p-x[j]
        d+=volume*fluid.kernel(r,h)/norm;g=volume*fluid.grad(r,h,norm)
        gsum+=g
        if j!=i:g2+=wp.dot(g,g)
        r=p-wp.vec3(x[j][0],x[j][1],-x[j][2])
        d+=volume*fluid.kernel(r,h)/norm;gsum+=volume*fluid.grad(r,h,norm)
    rho[i]=d;lam[i]=0.
    if groups[i]<0:lam[i]=-wp.max(d-1.,0.)/(g2+wp.dot(gsum,gsum)+.02/(h*h))


@wp.kernel(enable_backward=False)
def plate_corrections(x:wp.array(dtype=wp.vec3),e:wp.array(dtype=float),groups:wp.array(dtype=int),
                      grid:wp.uint64,h:float,norm:float,dx:float,lam:wp.array(dtype=float),
                      delta:wp.array(dtype=wp.vec3)):
    i=wp.tid();d=delta[i]
    if groups[i]<0:
        for j in wp.hash_grid_query(grid,x[i],h):
            r=x[i]-wp.vec3(x[j][0],x[j][1],-x[j][2])
            d+=lam[i]*(1.-.083*thermal.liquid(e[j]))*fluid.grad(r,h,norm)
        if wp.length(d)>dx*.35:d*=dx*.35/wp.length(d)
        delta[i]=d


@wp.func
def color_gradient(r:wp.vec3,h:float,norm:float):
    q=wp.max(1.-wp.dot(r,r)/(h*h),0.)
    return -6.*q*q*r/(h*h*norm)


@wp.kernel(enable_backward=False)
def normals(x:wp.array(dtype=wp.vec3),e:wp.array(dtype=float),grid:wp.uint64,
            h:float,norm:float,mirror:int,out:wp.array(dtype=wp.vec3)):
    i=wp.tid();p=x[i];n=wp.vec3(0.)
    for j in wp.hash_grid_query(grid,p,h):
        volume=1.-.083*thermal.liquid(e[j])
        n+=volume*color_gradient(p-x[j],h,norm)
        if mirror!=0:
            reflected=wp.vec3(x[j][0],x[j][1],-x[j][2])
            n+=volume*color_gradient(p-reflected,h,norm)
    out[i]=n


@wp.kernel(enable_backward=False)
def accelerate(x:wp.array(dtype=wp.vec3),v:wp.array(dtype=wp.vec3),e:wp.array(dtype=float),
               groups:wp.array(dtype=int),n:wp.array(dtype=wp.vec3),grid:wp.uint64,
               h:float,norm:float,sigma:float,dt:float,mirror:int):
    i=wp.tid();ni=n[i];length=wp.length(ni)
    if groups[i]<0 and length>.05/h:
        ni/=length;kappa=float(0.);weight=float(0.)
        for j in wp.hash_grid_query(grid,x[i],h):
            nj=n[j];nj_length=wp.length(nj)
            if nj_length>.05/h:
                nj/=nj_length;volume=1.-.083*thermal.liquid(e[j])
                r=x[i]-x[j];q=wp.max(1.-wp.dot(r,r)/(h*h),0.)
                kappa-=volume*wp.dot(nj-ni,color_gradient(r,h,norm))
                weight+=volume*q*q*q/norm
                if mirror!=0:
                    r=x[i]-wp.vec3(x[j][0],x[j][1],-x[j][2])
                    reflected=wp.vec3(nj[0],nj[1],-nj[2])
                    q=wp.max(1.-wp.dot(r,r)/(h*h),0.)
                    kappa-=volume*wp.dot(reflected-ni,color_gradient(r,h,norm))
                    weight+=volume*q*q*q/norm
        kappa=wp.clamp(kappa/wp.max(weight,.25),-2./h,2./h)
        # Color gradient supplies the regularized surface delta in the CSF law.
        v[i]+=dt*(sigma/1000.)*kappa*n[i]
