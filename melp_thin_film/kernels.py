"""Original Warp implementation of conservative two-layer thin-film dynamics.

MELP reference: Deng et al., DOI 10.1145/3528223.3530174, Eqs. 6--21.
Wendland kernels, a symmetric graph discretization and implicit normal dynamics
are numerical adaptations; geometry and material particles remain independent.
"""
import warp as wp
D=wp.float64
V=wp.vec3d
M=wp.mat33d
U6=wp.types.vector(length=6,dtype=D)
M6=wp.types.matrix(shape=(6,6),dtype=D)
PI=wp.constant(D(3.141592653589793))

@wp.func
def tangent(v:V,n:V):
    return v-wp.dot(v,n)*n

@wp.func
def axis(n:V):
    a=V(D(0),D(0),D(1))
    if wp.abs(n[2])>D(.85):a=V(D(0),D(1),D(0))
    return wp.normalize(wp.cross(a,n))

@wp.func
def weight(r:D,h:D):
    q=r/h
    v=wp.max(D(0),D(1)-q)
    return v*v*v*v*(D(1)+D(4)*q)

@wp.func
def grad_scale(r:D,h:D):
    q=r/h
    v=wp.max(D(0),D(1)-q)
    return D(20)*v*v*v/(h*h)

@wp.func
def fit_coeff(q:V,n:V,x:wp.array(dtype=V),neighbors:wp.array2d(dtype=int),i:int,count:int,h:D):
    e1=axis(n);e2=wp.cross(n,e1)
    mat=M6();rhs=U6()
    for k in range(count):
        j=neighbors[i,k];dx=(x[j]-q)/h
        u=wp.dot(dx,e1);v=wp.dot(dx,e2);z=wp.dot(dx,n)
        w=weight(wp.length(x[j]-q),h)
        f=U6(D(1),u,v,u*u,u*v,v*v)
        for a in range(6):
            rhs[a]=rhs[a]+w*f[a]*z
            for b in range(6):mat[a,b]=mat[a,b]+w*f[a]*f[b]
    for a in range(6):mat[a,a]=mat[a,a]+D(1.e-9)
    # Cholesky factorization of the weighted local quadratic fit.
    L=M6()
    for a in range(6):
        for b in range(a+1):
            s=mat[a,b]
            for k in range(b):s=s-L[a,k]*L[b,k]
            if a==b:L[a,b]=wp.sqrt(wp.max(s,D(1.e-14)))
            else:L[a,b]=s/L[b,b]
    y=U6();c=U6()
    for a in range(6):
        s=rhs[a]
        for k in range(a):s=s-L[a,k]*y[k]
        y[a]=s/L[a,a]
    for aa in range(6):
        a=5-aa;s=y[a]
        for k in range(a+1,6):s=s-L[k,a]*c[k]
        c[a]=s/L[a,a]
    return c

@wp.kernel
def float_positions(x:wp.array(dtype=V),out:wp.array(dtype=wp.vec3)):
    i=wp.tid();out[i]=wp.vec3(x[i])

@wp.kernel
def find_neighbors(grid:wp.uint64,x:wp.array(dtype=V),groups:wp.array(dtype=int),
                   query:wp.array(dtype=V),query_groups:wp.array(dtype=int),h:D,
                   nb:wp.array2d(dtype=int),counts:wp.array(dtype=int),capacity:int):
    i=wp.tid();count=int(0)
    for j in wp.hash_grid_query(grid,wp.vec3(query[i]),float(h)):
        if groups[j]==query_groups[i] and wp.length(x[j]-query[i])<h:
            if count<capacity:nb[i,count]=j
            count+=1
    counts[i]=count

@wp.kernel
def surface_geometry(x:wp.array(dtype=V),n:wp.array(dtype=V),nb:wp.array2d(dtype=int),count:wp.array(dtype=int),h:D,
                     area:wp.array(dtype=D),newn:wp.array(dtype=V),mls_curvature:wp.array(dtype=D),density_gradient:wp.array(dtype=V)):
    i=wp.tid();normal=n[i];e1=axis(normal);e2=wp.cross(normal,e1)
    c=fit_coeff(x[i],normal,x,nb,i,count[i],h)
    normal=wp.normalize(normal-c[1]*e1-c[2]*e2)
    newn[i]=normal
    den=D(0);gd=V()
    for k in range(count[i]):
        j=nb[i,k];dx=x[j]-x[i];dist=wp.length(dx)
        den+=weight(dist,h)
        gd+=grad_scale(dist,h)*tangent(dx,normal)
    normalization=D(7)/(PI*h*h)
    area[i]=D(1)/(den*normalization)
    density_gradient[i]=gd/wp.max(den,D(1.e-20))
    mls_curvature[i]=-D(2)*(c[3]+c[5])/h

@wp.kernel
def interpolate_weights(x:wp.array(dtype=V),n:wp.array(dtype=V),p:wp.array(dtype=V),nb:wp.array2d(dtype=int),counts:wp.array(dtype=int),h:D,
                        weights:wp.array2d(dtype=D),lp_n:wp.array(dtype=V),mean_offset:wp.array(dtype=V)):
    i=wp.tid();total=D(0);nn=V();mean=V()
    for k in range(counts[i]):
        j=nb[i,k];w=weight(wp.length(p[i]-x[j]),h);total+=w;nn+=w*n[j];mean+=w*(x[j]-p[i])
    lp_n[i]=wp.normalize(nn);mean_offset[i]=mean/wp.max(total,D(1.e-20))
    for k in range(counts[i]):weights[i,k]=weight(wp.length(p[i]-x[nb[i,k]]),h)/wp.max(total,D(1.e-20))

@wp.kernel
def scatter(x:wp.array(dtype=V),p:wp.array(dtype=V),v:wp.array(dtype=V),affine:wp.array(dtype=M),normal:wp.array(dtype=V),
            mean:wp.array(dtype=V),vol:wp.array(dtype=D),soap:wp.array(dtype=D),nb:wp.array2d(dtype=int),count:wp.array(dtype=int),
            weights:wp.array2d(dtype=D),ev:wp.array(dtype=D),ec:wp.array(dtype=D),momentum:wp.array(dtype=V)):
    i=wp.tid()
    for k in range(count[i]):
        j=nb[i,k];w=weights[i,k]
        # Center the affine offsets to retain exact partition-of-unity momentum.
        dv=affine[i]*tangent(x[j]-p[i]-mean[i],normal[i])
        wp.atomic_add(ev,j,w*vol[i]);wp.atomic_add(ec,j,w*soap[i])
        wp.atomic_add(momentum,j,w*vol[i]*(v[i]+dv))

@wp.kernel
def collect(ev:wp.array(dtype=D),ec:wp.array(dtype=D),momentum:wp.array(dtype=V),area:wp.array(dtype=D),
            thickness:wp.array(dtype=D),gamma:wp.array(dtype=D),velocity:wp.array(dtype=V)):
    i=wp.tid();thickness[i]=ev[i]/area[i];gamma[i]=ec[i]/area[i]
    velocity[i]=momentum[i]/wp.max(ev[i],D(1.e-30))

@wp.kernel
def graph(x:wp.array(dtype=V),n:wp.array(dtype=V),area:wp.array(dtype=D),nb:wp.array2d(dtype=int),count:wp.array(dtype=int),h:D,
          lap:wp.array2d(dtype=D),gradient:wp.array2d(dtype=V)):
    i=wp.tid();cov=M();normal=n[i]
    for k in range(count[i]):
        j=nb[i,k];dx=tangent(x[j]-x[i],normal);r=wp.length(x[j]-x[i])
        w=grad_scale(r,h)*D(7)/(PI*h*h)
        cov+=area[j]*w*wp.outer(dx,dx)
    cov+=wp.outer(normal,normal)
    inv=wp.inverse(cov+D(1.e-10)*M(D(1),D(0),D(0),D(0),D(1),D(0),D(0),D(0),D(1)))
    for k in range(count[i]):
        j=nb[i,k];dx=x[j]-x[i];r=wp.length(dx)
        w=grad_scale(r,h)*D(7)/(PI*h*h)
        lap[i,k]=D(2)*area[i]*area[j]*w
        gradient[i,k]=area[j]*(inv*(w*tangent(dx,normal)))

@wp.kernel
def statistics(x:wp.array(dtype=V),n:wp.array(dtype=V),area:wp.array(dtype=D),group:wp.array(dtype=int),centers:wp.array(dtype=V),
               volume:wp.array(dtype=D),surface:wp.array(dtype=D)):
    i=wp.tid();g=group[i]
    wp.atomic_add(volume,g,area[i]*wp.dot(x[i]-centers[g],n[i])/D(3))
    wp.atomic_add(surface,g,area[i])

@wp.kernel
def invert_diagonal(diag:wp.array(dtype=D),inv:wp.array(dtype=D)):
    i=wp.tid();inv[i]=D(1)/diag[i]

@wp.kernel
def update_tangential(v:wp.array(dtype=V),normal:wp.array(dtype=V),gamma:wp.array(dtype=D),thick:wp.array(dtype=D),nb:wp.array2d(dtype=int),
                      count:wp.array(dtype=int),grad:wp.array2d(dtype=V),gravity:V,dt:D,rt:D,rho:D,gravity_scale:D,damping:D,out:wp.array(dtype=V)):
    i=wp.tid();g=V()
    for k in range(count[i]):
        j=nb[i,k];g+=(gamma[j]-gamma[i])*grad[i,k]
    out[i]=tangent(v[i]+dt*gravity_scale*gravity,normal[i])-dt*D(2)*rt/(rho*thick[i])*g
    out[i]*=wp.exp(-dt*damping)

@wp.kernel
def normal_rhs(x:wp.array(dtype=V),n:wp.array(dtype=V),v:wp.array(dtype=V),area:wp.array(dtype=D),ev:wp.array(dtype=D),gamma:wp.array(dtype=D),
               nb:wp.array2d(dtype=int),count:wp.array(dtype=int),lap:wp.array2d(dtype=D),group:wp.array(dtype=int),
               volume:wp.array(dtype=D),gas:wp.array(dtype=D),dt:D,rho:D,rt:D,sigma0:D,atmosphere:D,damping:D,added_mass:D,
               internal:wp.array(dtype=D),force:wp.array(dtype=D),share:wp.array(dtype=D),preserve_contact_traction:D,diag:wp.array(dtype=D),rhs:wp.array(dtype=D),coeff:wp.array2d(dtype=D),inv:wp.array(dtype=D)):
    i=wp.tid();mass=rho*ev[i]+added_mass*area[i];g=group[i];cap=V();isolated_cap=V();total=D(0)
    for k in range(count[i]):
        j=nb[i,k];sig=D(2)*sigma0-rt*(gamma[i]+gamma[j])
        c=sig*(D(1)-D(.25)*(share[i]+share[j]))*lap[i,k]
        if i==j:c=D(0)
        cap+=c*(x[j]-x[i]);total+=c
        isolated_cap+=sig*lap[i,k]*(x[j]-x[i])
        coeff[i,k]=dt*dt*c*wp.dot(n[i],n[j])
    pin=gas[g]/volume[g]
    diag[i]=mass*(D(1)+dt*damping)+dt*dt*total
    # The prescribed quadrupole is a zero-resultant shape actuator. Include it
    # in the discrete force-balance projection to prevent sampling-induced drift.
    applied=wp.dot(cap,n[i])+(pin-atmosphere+force[i])*area[i]
    # Closed isolated-film quadrature error has zero physical resultant.
    # The traction difference caused by a shared lamella is a real inter-region
    # force and must not be removed separately from each bubble.
    balanced_cap=(D(1)-preserve_contact_traction)*cap+preserve_contact_traction*isolated_cap
    internal[i]=wp.dot(balanced_cap,n[i])+(pin-atmosphere+force[i])*area[i]
    rhs[i]=mass*wp.dot(v[i],n[i])+dt*applied
    inv[i]=D(1)/(diag[i]+dt*dt*pin/volume[g]*area[i]*area[i])

@wp.kernel
def rank_reduction(area:wp.array(dtype=D),group:wp.array(dtype=int),x:wp.array(dtype=D),sum:wp.array(dtype=D)):
    i=wp.tid();wp.atomic_add(sum,group[i],area[i]*x[i])

@wp.kernel
def normal_matvec(nb:wp.array2d(dtype=int),count:wp.array(dtype=int),coeff:wp.array2d(dtype=D),diag:wp.array(dtype=D),
                  area:wp.array(dtype=D),group:wp.array(dtype=int),gas:wp.array(dtype=D),volume:wp.array(dtype=D),sum:wp.array(dtype=D),dt:D,
                  x:wp.array(dtype=D),y:wp.array(dtype=D),z:wp.array(dtype=D),alpha:D,beta:D):
    i=wp.tid();value=diag[i]*x[i]
    for k in range(count[i]):value-=coeff[i,k]*x[nb[i,k]]
    g=group[i];value+=dt*dt*gas[g]/(volume[g]*volume[g])*area[i]*sum[g]
    z[i]=alpha*value+beta*y[i]

@wp.kernel
def combine(vt:wp.array(dtype=V),vn:wp.array(dtype=D),normal:wp.array(dtype=V),v:wp.array(dtype=V)):
    i=wp.tid();v[i]=vt[i]+vn[i]*normal[i]

@wp.kernel
def e_to_l(x:wp.array(dtype=V),p:wp.array(dtype=V),normal:wp.array(dtype=V),ev:wp.array(dtype=V),
           nb:wp.array2d(dtype=int),count:wp.array(dtype=int),weights:wp.array2d(dtype=D),mean:wp.array(dtype=V),
           v:wp.array(dtype=V),affine:wp.array(dtype=M),affine_strength:D):
    i=wp.tid();vel=V();B=M();cov=M();n=normal[i]
    for k in range(count[i]):vel+=weights[i,k]*ev[nb[i,k]]
    for k in range(count[i]):
        j=nb[i,k];dx=tangent(x[j]-p[i]-mean[i],n);w=weights[i,k]
        B+=w*wp.outer(ev[j]-vel,dx);cov+=w*wp.outer(dx,dx)
    cov+=wp.outer(n,n)
    v[i]=vel
    affine[i]=affine_strength*B*wp.inverse(cov+D(1.e-14)*M(D(1),D(0),D(0),D(0),D(1),D(0),D(0),D(0),D(1)))

@wp.kernel
def advance_e(x:wp.array(dtype=V),n:wp.array(dtype=V),vn:wp.array(dtype=D),density_gradient:wp.array(dtype=V),dt:D,h:D,redistribute:D):
    i=wp.tid();shift=-redistribute*h*h*density_gradient[i]
    length=wp.length(shift)
    if length>D(.02)*h:shift*=D(.02)*h/length
    x[i]+=dt*vn[i]*n[i]+shift

@wp.kernel
def advance_l(x:wp.array(dtype=V),v:wp.array(dtype=V),dt:D):
    i=wp.tid();x[i]+=dt*v[i]

@wp.kernel
def redistribute_surface(x:wp.array(dtype=V),n:wp.array(dtype=V),nb:wp.array2d(dtype=int),count:wp.array(dtype=int),
                         gradient:wp.array(dtype=V),other:wp.array(dtype=int),share:wp.array(dtype=D),contact_aware:D,h:D,strength:D,out:wp.array(dtype=V)):
    i=wp.tid();shift=-strength*h*h*gradient[i];length=wp.length(shift)
    if length>D(.04)*h:shift*=D(.04)*h/length
    j=other[i]
    if j>=0 and contact_aware>D(0) and share[i]>D(.1):
        cross_normal=tangent(n[j],n[i]);square=wp.length_sq(cross_normal)
        if square>D(.01):shift-=share[i]*wp.dot(shift,cross_normal)/square*cross_normal
    c=fit_coeff(x[i],n[i],x,nb,i,count[i],h);e1=axis(n[i]);e2=wp.cross(n[i],e1)
    u=wp.dot(shift,e1)/h;v=wp.dot(shift,e2)/h
    # Anchor the quadratic to the original sample. Omitting c[0] prevents a
    # repeated MLS fit bias from shrinking the interface during redistribution.
    height=h*(c[1]*u+c[2]*v+c[3]*u*u+c[4]*u*v+c[5]*v*v)
    out[i]=x[i]+shift+height*n[i]

@wp.func
def one_sided_capillary(i:int,x:wp.array(dtype=V),area:wp.array(dtype=D),gamma:wp.array(dtype=D),nb:wp.array2d(dtype=int),count:wp.array(dtype=int),lap:wp.array2d(dtype=D),sigma0:D,rt:D):
    cap=V()
    for k in range(count[i]):
        j=nb[i,k];sig=sigma0-D(.5)*rt*(gamma[i]+gamma[j]);cap+=sig*lap[i,k]*(x[j]-x[i])
    return cap/area[i]

@wp.kernel
def capillary_field(x:wp.array(dtype=V),area:wp.array(dtype=D),gamma:wp.array(dtype=D),nb:wp.array2d(dtype=int),count:wp.array(dtype=int),lap:wp.array2d(dtype=D),sigma0:D,rt:D,out:wp.array(dtype=V)):
    i=wp.tid();out[i]=one_sided_capillary(i,x,area,gamma,nb,count,lap,sigma0,rt)

@wp.kernel
def multi_region_traction(grid:wp.uint64,x:wp.array(dtype=V),n:wp.array(dtype=V),area:wp.array(dtype=D),gamma:wp.array(dtype=D),
                          capillary:wp.array(dtype=V),other:wp.array(dtype=int),share:wp.array(dtype=D),cross_nb:wp.array2d(dtype=int),cross_count:wp.array(dtype=int),cross_w:wp.array2d(dtype=D),continuous:int,
                          group:wp.array(dtype=int),gas:wp.array(dtype=D),volume:wp.array(dtype=D),sigma0:D,rt:D,atmosphere:D,h:D,dt:D,rhs:wp.array(dtype=D)):
    i=wp.tid();j=other[i]
    if j<0:return
    coupling=share[i]
    ci=capillary[i];cj=capillary[j];guest_n=n[j]
    if continuous!=0:
        cj=V();guest_n=V()
        for k in range(cross_count[i]):
            node=cross_nb[i,k];w=cross_w[i,k];cj+=w*capillary[node];guest_n+=w*n[node]
    pi=gas[group[i]]/volume[group[i]]-atmosphere;pj=gas[group[j]]/volume[group[j]]-atmosphere
    ambient_n=(n[i]+coupling*guest_n)/(D(1)+coupling);length=wp.length(ambient_n);outer=V()
    if length>D(.025):
        unit=ambient_n/length;curvature=D(0)
        for k in wp.hash_grid_query(grid,wp.vec3(x[i]),float(h)):
            exterior=(D(1)-share[k])*(D(1)-share[k])
            if group[k]!=group[i]:exterior*=coupling
            dx=x[k]-x[i];r=wp.length(dx)
            if r<h:curvature+=exterior*D(2)*area[k]*grad_scale(r,h)*D(7)/(PI*h*h)*wp.dot(dx,unit)
        outer=(sigma0-rt*gamma[i])*curvature*ambient_n
    # Opposing enclosed pressures cancel on a shared film; the remaining outer
    # interface near its rim supplies the junction traction (paper Eqs. 23–26).
    shared=(ci+pi*n[i]+coupling*(cj+pj*guest_n)+outer)/(D(1)+coupling)
    isolated=D(2)*ci+pi*n[i]
    rhs[i]+=coupling*dt*area[i]*wp.dot(shared-isolated,n[i])

@wp.kernel
def smooth_normal(x:wp.array(dtype=V),n:wp.array(dtype=V),area:wp.array(dtype=D),v:wp.array(dtype=V),
                  nb:wp.array2d(dtype=int),count:wp.array(dtype=int),h:D,amount:D,out:wp.array(dtype=D)):
    i=wp.tid();total=D(0);mean=V()
    for k in range(count[i]):
        j=nb[i,k];w=area[j]*weight(wp.length(x[j]-x[i]),h);total+=w;mean+=w*v[j]
    out[i]=wp.dot((D(1)-amount)*v[i]+amount*mean/total,n[i])

@wp.kernel
def replace_normal(v:wp.array(dtype=V),n:wp.array(dtype=V),vn:wp.array(dtype=D)):
    i=wp.tid();v[i]=tangent(v[i],n[i])+vn[i]*n[i]

@wp.kernel
def continuous_contacts(grid:wp.uint64,x:wp.array(dtype=V),n:wp.array(dtype=V),groups:wp.array(dtype=int),h:D,capacity:int,
                        other:wp.array(dtype=int),directions:wp.array(dtype=V),gap:wp.array(dtype=D),share:wp.array(dtype=D),
                        nb:wp.array2d(dtype=int),count:wp.array(dtype=int),weights:wp.array2d(dtype=D)):
    i=wp.tid();best=h*h;nearest=int(-1);other[i]=-1;share[i]=D(0);gap[i]=D(0);count[i]=0
    for j in wp.hash_grid_query(grid,wp.vec3(x[i]),float(h)):
        if groups[j]!=groups[i] and wp.dot(n[i],n[j])<D(-.2):
            dist=wp.length_sq(x[j]-x[i])
            if dist<best:best=dist;nearest=j
    if nearest<0:return
    guest=groups[nearest];num=int(0);total=D(0);normal=V()
    for j in wp.hash_grid_query(grid,wp.vec3(x[i]),float(h)):
        if groups[j]==guest:
            r=wp.length(x[j]-x[i])
            if r<h:
                if num<capacity:
                    w=weight(r,h);nb[i,num]=j;weights[i,num]=w;total+=w;normal+=w*n[j]
                num+=1
    if num<8 or num>capacity:return
    normal=wp.normalize(normal)
    coeff=fit_coeff(x[i],normal,x,nb,i,num,h)
    e1=axis(normal);e2=wp.cross(normal,e1);projected_normal=wp.normalize(normal-coeff[1]*e1-coeff[2]*e2)
    nc=wp.normalize(n[i]-projected_normal);signed_gap=-h*coeff[0]*wp.dot(normal,nc)
    if signed_gap<=-D(.22)*h:return
    other[i]=nearest;count[i]=num;directions[i]=nc;gap[i]=signed_gap
    t=wp.clamp(D(1)+signed_gap/(D(.22)*h),D(0),D(1));share[i]=t*t*(D(3)-D(2)*t)
    for k in range(num):weights[i,k]/=total

@wp.kernel
def continuous_contact_rhs(other:wp.array(dtype=int),nb:wp.array2d(dtype=int),count:wp.array(dtype=int),weights:wp.array2d(dtype=D),
                           directions:wp.array(dtype=V),gap:wp.array(dtype=D),share:wp.array(dtype=D),n:wp.array(dtype=V),area:wp.array(dtype=D),
                           dt:D,stiffness:D,separation:D,adhesion:int,diag:wp.array(dtype=D),rhs:wp.array(dtype=D),strength:wp.array(dtype=D)):
    i=wp.tid();strength[i]=D(0)
    if other[i]<0:return
    g=gap[i]+separation
    if adhesion==0 and g<=D(0):return
    c=stiffness*area[i]*share[i];strength[i]=c;J=wp.dot(n[i],directions[i])
    wp.atomic_add(rhs,i,-dt*c*g*J);wp.atomic_add(diag,i,dt*dt*c*J*J)
    for k in range(count[i]):
        j=nb[i,k];Jj=-weights[i,k]*wp.dot(n[j],directions[i])
        wp.atomic_add(rhs,j,-dt*c*g*Jj);wp.atomic_add(diag,j,dt*dt*c*Jj*Jj)

@wp.kernel
def continuous_contact_matvec(other:wp.array(dtype=int),nb:wp.array2d(dtype=int),count:wp.array(dtype=int),weights:wp.array2d(dtype=D),
                              directions:wp.array(dtype=V),n:wp.array(dtype=V),strength:wp.array(dtype=D),x:wp.array(dtype=D),out:wp.array(dtype=D),dt:D,alpha:D):
    i=wp.tid()
    if other[i]<0:return
    J=wp.dot(n[i],directions[i]);value=J*x[i]
    for k in range(count[i]):
        j=nb[i,k];value-=weights[i,k]*wp.dot(n[j],directions[i])*x[j]
    value*=alpha*dt*dt*strength[i];wp.atomic_add(out,i,J*value)
    for k in range(count[i]):
        j=nb[i,k];wp.atomic_add(out,j,-weights[i,k]*wp.dot(n[j],directions[i])*value)

@wp.kernel
def contact_preconditioner(contact_diag:wp.array(dtype=D),inverse:wp.array(dtype=D)):
    i=wp.tid();inverse[i]=D(1)/(D(1)/inverse[i]+contact_diag[i])

@wp.kernel
def project_l(x:wp.array(dtype=V),n:wp.array(dtype=V),p:wp.array(dtype=V),nb:wp.array2d(dtype=int),count:wp.array(dtype=int),
              h:D,normal:wp.array(dtype=V),correction:wp.array(dtype=D)):
    i=wp.tid();nn=V()
    for k in range(count[i]):
        j=nb[i,k];nn+=weight(wp.length(p[i]-x[j]),h)*n[j]
    nn=wp.normalize(nn)
    c=fit_coeff(p[i],nn,x,nb,i,count[i],h)
    p[i]+=h*c[0]*nn
    e1=axis(nn);e2=wp.cross(nn,e1)
    normal[i]=wp.normalize(nn-c[1]*e1-c[2]*e2);correction[i]=wp.abs(h*c[0])

@wp.kernel
def dye_thickness(x:wp.array(dtype=V),n:wp.array(dtype=V),v:wp.array(dtype=V),p:wp.array(dtype=V),lv:wp.array(dtype=V),
                  nb:wp.array2d(dtype=int),count:wp.array(dtype=int),weights:wp.array2d(dtype=D),
                  grad:wp.array2d(dtype=V),enb:wp.array2d(dtype=int),ecount:wp.array(dtype=int),divergence:wp.array(dtype=D)):
    i=wp.tid();dv=D(0)
    for k in range(ecount[i]):
        j=enb[i,k];dv+=wp.dot(v[j]-v[i],grad[i,k])
    divergence[i]=dv

@wp.kernel
def advect_thickness(nb:wp.array2d(dtype=int),count:wp.array(dtype=int),weights:wp.array2d(dtype=D),div:wp.array(dtype=D),lh:wp.array(dtype=D),dt:D):
    i=wp.tid();dv=D(0)
    for k in range(count[i]):dv+=weights[i,k]*div[nb[i,k]]
    lh[i]*=wp.exp(-dt*dv)

@wp.kernel
def cross_contacts(grid:wp.uint64,x:wp.array(dtype=V),n:wp.array(dtype=V),groups:wp.array(dtype=int),h:D,
                   other:wp.array(dtype=int),directions:wp.array(dtype=V),gap:wp.array(dtype=D),share:wp.array(dtype=D)):
    i=wp.tid();best=h*h;chosen=int(-1)
    for j in wp.hash_grid_query(grid,wp.vec3(x[i]),float(h)):
        if groups[j]!=groups[i] and wp.dot(n[i],n[j])<D(-.2):
            d=wp.length_sq(x[j]-x[i])
            if d<best:best=d;chosen=j
    other[i]=-1;share[i]=D(0);gap[i]=D(0)
    if chosen>=0:
        nc=wp.normalize(n[i]-n[chosen]);g=wp.dot(x[i]-x[chosen],nc)
        if g>-D(.22)*h:
            other[i]=chosen;directions[i]=nc;gap[i]=g
            t=wp.clamp(D(1)+g/(D(.22)*h),D(0),D(1))
            share[i]=t*t*(D(3)-D(2)*t)

@wp.kernel
def contact_rhs(other:wp.array(dtype=int),directions:wp.array(dtype=V),gap:wp.array(dtype=D),share:wp.array(dtype=D),n:wp.array(dtype=V),
                area:wp.array(dtype=D),dt:D,stiffness:D,separation:D,diag:wp.array(dtype=D),rhs:wp.array(dtype=D),strength:wp.array(dtype=D)):
    i=wp.tid();j=other[i];strength[i]=D(0)
    if j>=0:
        c=stiffness*D(.5)*(area[i]+area[j])*share[i]
        strength[i]=c;ni=wp.dot(n[i],directions[i]);nj=-wp.dot(n[j],directions[i]);g=gap[i]+separation
        wp.atomic_add(rhs,i,-dt*c*g*ni);wp.atomic_add(rhs,j,-dt*c*g*nj)
        wp.atomic_add(diag,i,dt*dt*c*ni*ni);wp.atomic_add(diag,j,dt*dt*c*nj*nj)

@wp.kernel
def contact_matvec(other:wp.array(dtype=int),directions:wp.array(dtype=V),n:wp.array(dtype=V),strength:wp.array(dtype=D),
                   x:wp.array(dtype=D),z:wp.array(dtype=D),dt:D,alpha:D):
    i=wp.tid();j=other[i]
    if j>=0:
        ni=wp.dot(n[i],directions[i]);nj=-wp.dot(n[j],directions[i]);v=alpha*dt*dt*strength[i]*(ni*x[i]+nj*x[j])
        wp.atomic_add(z,i,ni*v);wp.atomic_add(z,j,nj*v)

@wp.kernel
def drive_pressure(x:wp.array(dtype=V),n:wp.array(dtype=V),group:wp.array(dtype=int),centers:wp.array(dtype=V),
                   time:D,amplitude:D,out:wp.array(dtype=D)):
    i=wp.tid();v=wp.normalize(x[i]-centers[group[i]])
    a=V(wp.cos(D(4)*time),wp.sin(D(4)*time),D(.25));a=wp.normalize(a)
    envelope=(D(1)-wp.exp(-D(20)*time))*wp.sin(D(22)*time)
    out[i]=amplitude*envelope*(D(3)*wp.dot(v,a)*wp.dot(v,a)-D(1))

@wp.kernel
def compatible_init(gamma:wp.array(dtype=D),area:wp.array(dtype=D),base:wp.array(dtype=D),diag:wp.array(dtype=D),rhs:wp.array(dtype=D)):
    i=wp.tid();b=area[i]/gamma[i];base[i]=b;diag[i]=b;rhs[i]=area[i]

@wp.kernel
def compatible_rhs(v:wp.array(dtype=V),n:wp.array(dtype=V),area:wp.array(dtype=D),thick:wp.array(dtype=D),
                   nb:wp.array2d(dtype=int),count:wp.array(dtype=int),grad:wp.array2d(dtype=V),
                   gravity:V,dt:D,rt:D,rho:D,gravity_scale:D,diag:wp.array(dtype=D),rhs:wp.array(dtype=D)):
    i=wp.tid();u=tangent(v[i]+dt*gravity_scale*gravity,n[i]);gs=V();co=dt*dt*D(2)*rt/rho*area[i]/thick[i]
    for k in range(count[i]):
        j=nb[i,k]
        if j!=i:
            g=grad[i,k];gs+=g
            wp.atomic_add(diag,j,co*wp.dot(g,g))
            f=dt*area[i]*wp.dot(g,u);wp.atomic_add(rhs,j,f);wp.atomic_add(rhs,i,-f)
    wp.atomic_add(diag,i,co*wp.dot(gs,gs))

@wp.kernel
def compatible_gradient(x:wp.array(dtype=D),area:wp.array(dtype=D),thick:wp.array(dtype=D),
                        nb:wp.array2d(dtype=int),count:wp.array(dtype=int),grad:wp.array2d(dtype=V),
                        rt:D,rho:D,tmp:wp.array(dtype=V)):
    i=wp.tid();g=V()
    for k in range(count[i]):g+=(x[nb[i,k]]-x[i])*grad[i,k]
    tmp[i]=D(2)*rt/rho*area[i]/thick[i]*g

@wp.kernel
def compatible_base(base:wp.array(dtype=D),x:wp.array(dtype=D),y:wp.array(dtype=D),z:wp.array(dtype=D),alpha:D,beta:D):
    i=wp.tid();z[i]=alpha*base[i]*x[i]+beta*y[i]

@wp.kernel
def compatible_adjoint(tmp:wp.array(dtype=V),nb:wp.array2d(dtype=int),count:wp.array(dtype=int),grad:wp.array2d(dtype=V),z:wp.array(dtype=D),dt:D,alpha:D):
    i=wp.tid()
    for k in range(count[i]):
        j=nb[i,k];f=alpha*dt*dt*wp.dot(grad[i,k],tmp[i]);wp.atomic_add(z,j,f);wp.atomic_add(z,i,-f)

@wp.kernel
def use_geometric_normal(v:wp.array(dtype=V),n:wp.array(dtype=V),nv:wp.array(dtype=V)):
    i=wp.tid();v[i]=tangent(v[i],n[i])+wp.dot(nv[i],n[i])*n[i]

@wp.kernel
def remember_normal(vn:wp.array(dtype=D),n:wp.array(dtype=V),out:wp.array(dtype=V)):
    i=wp.tid();out[i]=vn[i]*n[i]

@wp.kernel
def force_balance_reduce(n:wp.array(dtype=V),area:wp.array(dtype=D),group:wp.array(dtype=int),internal:wp.array(dtype=D),
                         net:wp.array(dtype=V),tensor:wp.array2d(dtype=D)):
    i=wp.tid();g=group[i];wp.atomic_add(net,g,internal[i]*n[i])
    for a in range(3):
        for b in range(3):wp.atomic_add(tensor,g,a*3+b,area[i]*n[i][a]*n[i][b])

@wp.kernel
def force_balance_apply(n:wp.array(dtype=V),area:wp.array(dtype=D),group:wp.array(dtype=int),net:wp.array(dtype=V),tensor:wp.array2d(dtype=D),dt:D,rhs:wp.array(dtype=D)):
    i=wp.tid();g=group[i];mat=M()
    for a in range(3):
        for b in range(3):mat[a,b]=tensor[g,a*3+b]
    correction=wp.inverse(mat)*net[g]
    rhs[i]-=dt*area[i]*wp.dot(n[i],correction)
