"""Projected Newton driver for the independently implemented Warp potentials.

Warp evaluates energies, exact reverse-mode gradients, numerical element Hessians,
magnetization, and state updates. NumPy/SciPy factor the small assembled Newton
matrix on the host. The same nonlinear solve includes contact and magnetism.
"""
import time
import numpy as np
import scipy.linalg
import warp as wp
from . import kernels as K


def rotate(q, v):
    t = 2*np.cross(q[..., :3], v)
    return v + q[..., 3:4]*t + np.cross(q[..., :3], t)


def multiply(a, b):
    return np.concatenate((a[:3]*b[3]+b[:3]*a[3]+np.cross(a[:3],b[:3]),
                           np.array([a[3]*b[3]-np.dot(a[:3],b[:3])])), axis=0)


def term(kind, a, b=0, **kw):
    t = K.Term()
    t.kind, t.a, t.b = kind, a, b
    for key, val in kw.items():
        setattr(t, key, K.V(*val) if key in {'ra','rb','ma','mb','normal','bias'} else float(val))
    return t


class Solver:
    def __init__(self, config, device='cuda:0'):
        wp.init()
        self.cfg = config
        self.device = device
        self.bodies = config['bodies']
        self.n = len(self.bodies)
        self.x = np.array([b['pos'] for b in self.bodies],float)
        self.q = np.array([b.get('quat',[0,0,0,1]) for b in self.bodies],float)
        self.v = np.zeros((self.n,3)); self.w = self.v.copy()
        self.mass = np.array([b['mass'] for b in self.bodies])
        self.I = np.array([b['inertia'] for b in self.bodies])
        self.free = np.array([b.get('free',[1]*6) for b in self.bodies],bool)
        self.sv = self.free[:,:3]/np.sqrt(self.mass[:,None])
        self.sw = self.free[:,3:]/np.sqrt(self.I.mean(1)[:,None])
        self.active = np.where(self.free.ravel())[0]
        self.damp = np.array([[b.get('damping',.08),b.get('angular_damping',.12),0] for b in self.bodies])
        self.samples = [(i,s) for i,b in enumerate(self.bodies) for s in b.get('samples',[])]
        self.so = np.array([i for i,s in self.samples],np.int32)
        self.sl = np.array([s['p'] for i,s in self.samples],float).reshape(-1,3)
        self.m = np.array([s.get('moment',[0,0,0]) for i,s in self.samples],float).reshape(-1,3)
        self.permanent = self.m.copy()
        self.response = np.array([4*np.pi/3*s['r']**3*s.get('chi',0)/(1+s.get('chi',0)/3) for i,s in self.samples])
        mi,mj=np.triu_indices(len(self.samples),1)
        mk=(self.so[mi]!=self.so[mj])&(np.any(self.free[self.so[mi]],axis=1)|np.any(self.free[self.so[mj]],axis=1))
        self.mi,self.mj=mi[mk],mj[mk]
        self.shapes = [(i,s) for i,b in enumerate(self.bodies) for s in b.get('shape',[])]
        self.co = np.array([i for i,s in self.shapes],np.int32)
        self.cl = np.array([s['p'] for i,s in self.shapes],float).reshape(-1,3)
        self.cr = np.array([s['r'] for i,s in self.shapes])
        aa,bb = np.triu_indices(len(self.shapes),1)
        keep = (self.co[aa] != self.co[bb]) & (np.any(self.free[self.co[aa]],axis=1)|np.any(self.free[self.co[bb]],axis=1))
        self.ca,self.cb = aa[keep],bb[keep]
        self.inside = np.array([self.bodies[i].get('inside',True) for i in self.co])
        self.kappa = config.get('contact_stiffness',1.e8)
        self.thickness = config.get('contact_thickness',.0015)
        self.clock = 0.
        self.history = []
        self.da = lambda a,dtype: wp.array(np.asarray(a),dtype=dtype,device=self.device)
        self.dsv,self.dsw = self.da(self.sv,K.V),self.da(self.sw,K.V)
        self.dactive = self.da(self.free.ravel().astype(np.int32),int)
        self.u = wp.zeros(self.n*6,dtype=K.D,device=device,requires_grad=True)
        self.E = wp.zeros(1,dtype=K.D,device=device,requires_grad=True)
        self.H = wp.zeros((self.n*6,self.n*6),dtype=K.D,device=device)
        self.damping = self.da(self.damp,K.V)
        self.so_w,self.sl_w = self.da(self.so,int),self.da(self.sl,K.V)

    def pose_numpy(self, u, dt):
        u = np.asarray(u).reshape(self.n,6)
        xx = self.base_x + dt*u[:,:3]*self.sv
        w = u[:,3:]*self.sw
        qq = np.empty_like(self.q)
        for i in range(self.n):
            d = np.r_[.5*dt*w[i],1.]
            qq[i] = multiply(d/np.linalg.norm(d),self.base_q[i])
        return xx,qq

    def centers(self, x, q):
        return x[self.co]+rotate(q[self.co],self.cl)

    def gaps(self, x, q):
        p = self.centers(x,q)
        gaps = [np.linalg.norm(p[self.ca]-p[self.cb],axis=1)-self.cr[self.ca]-self.cr[self.cb]]
        for plane in self.cfg.get('planes',[]):
            gaps.append((p@np.array(plane['normal'])-plane['offset']-self.cr)[self.inside])
        if 'container' in self.cfg:
            c = self.cfg['container']
            gaps.append((c['radius']-np.linalg.norm(p-np.array(c['center']),axis=1)-self.cr)[self.inside])
        return np.concatenate(gaps) if any(len(g) for g in gaps) else np.array([np.inf])

    def swept_safe(self, xa,qa,xb,qb):
        pa,pb = self.centers(xa,qa),self.centers(xb,qb)
        angles = 2*np.arccos(np.clip(np.abs(np.sum(qa*qb,axis=1)),0,1))
        curve = np.linalg.norm(self.cl,axis=1)*angles[self.co]**2/8
        a,b = self.ca,self.cb
        d0 = pa[a]-pa[b]; dd = pb[a]-pb[b]-d0
        tau = np.clip(-np.sum(d0*dd,axis=1)/np.maximum(np.sum(dd*dd,axis=1),1e-30),0,1)
        gap = np.linalg.norm(d0+tau[:,None]*dd,axis=1)-self.cr[a]-self.cr[b]-curve[a]-curve[b]
        if np.any(gap <= 1.e-8): return False
        for plane in self.cfg.get('planes',[]):
            n = np.array(plane['normal'])
            gap = np.minimum(pa@n,pb@n)-plane['offset']-self.cr-curve
            if np.any(gap[self.inside] <= 1.e-8): return False
        if 'container' in self.cfg:
            c = self.cfg['container']; center = np.array(c['center'])
            gap = c['radius']-np.maximum(np.linalg.norm(pa-center,axis=1),np.linalg.norm(pb-center,axis=1))-self.cr-curve
            if np.any(gap[self.inside] <= 1.e-8): return False
        return True

    def magnetize(self):
        if not len(self.samples): return
        # Synchronous, lagged magnetization as in the paper's Algorithm 1.
        old = self.m.copy(); old[self.response==0] = self.permanent[self.response==0]
        result = wp.empty(len(self.samples),dtype=K.V,device=self.device)
        wp.launch(K.induced_moments,len(self.samples),[self.so_w,self.sl_w,self.da(self.x,K.V),
                  self.da(self.q,K.Q),self.da(old,K.V),self.da(self.permanent,K.V),
                  self.da(self.response,K.D),len(self.samples),result],device=self.device)
        self.m = result.numpy()

    def make_terms(self, dt):
        terms = []
        p = self.x[self.so]+rotate(self.q[self.so],self.sl)
        strength=np.linalg.norm(self.m,axis=1)
        nonzero=strength[self.mi]*strength[self.mj]>1.e-18
        mi,mj=self.mi[nonzero],self.mj[nonzero]
        magnetic=np.zeros(len(mi),dtype=K.Term.numpy_dtype())
        magnetic['kind']=0;magnetic['a']=self.so[mi];magnetic['b']=self.so[mj]
        magnetic['ra']=self.sl[mi];magnetic['rb']=self.sl[mj]
        magnetic['ma']=self.m[mi];magnetic['mb']=self.m[mj]
        near=np.linalg.norm(p[mi]-p[mj],axis=1)<self.cfg.get('distance_threshold',1.e6)
        far=magnetic[~near];magnetic=magnetic[near]
        magnetic_count = len(magnetic)
        p = self.centers(self.x,self.q)
        # Broad phase includes a displacement margin. Acceptance checks below
        # reject any newly active pair omitted by this candidate set.
        a,b=self.ca,self.cb
        gap0=np.linalg.norm(p[a]-p[b],axis=1)-self.cr[a]-self.cr[b]
        margin=.04+2*dt*np.linalg.norm(self.v[self.co[a]]-self.v[self.co[b]],axis=1)
        self.contact_candidates=gap0<self.thickness+margin
        for ci,cj in zip(a[self.contact_candidates],b[self.contact_candidates]):
            a,b = int(self.co[ci]),int(self.co[cj])
            terms.append(term(1,a,b,ra=self.cl[ci],rb=self.cl[cj],radius=self.cr[ci]+self.cr[cj],
                              stiffness=self.kappa,thickness=self.thickness))
            dr = p[ci]-p[cj]; length=np.linalg.norm(dr); n=dr/length
            gap=length-self.cr[ci]-self.cr[cj]
            if gap<self.thickness:
                wa=p[ci]-self.cr[ci]*n; wb=p[cj]+self.cr[cj]*n
                self.add_friction(terms,a,b,wa,wb,n,gap,dt)
        for ci,(a,s) in enumerate(self.shapes):
            if not np.any(self.free[a]) or not self.inside[ci]: continue
            for plane in self.cfg.get('planes',[]):
                n=np.array(plane['normal']); offset=plane['offset']
                terms.append(term(2,a,ra=self.cl[ci],normal=n,radius=offset+self.cr[ci],stiffness=self.kappa,thickness=self.thickness))
                gap=np.dot(p[ci],n)-offset-self.cr[ci]
                if gap<self.thickness:
                    wa=p[ci]-self.cr[ci]*n; wb=wa-gap*n
                    self.add_friction(terms,a,0,wa,wb,n,gap,dt)
            if 'container' in self.cfg:
                c=self.cfg['container']; center=np.array(c['center'])
                terms.append(term(3,a,ra=self.cl[ci],rb=center,radius=c['radius']-self.cr[ci],stiffness=self.kappa,thickness=self.thickness))
                d=p[ci]-center; length=np.linalg.norm(d); n=-d/length
                gap=c['radius']-length-self.cr[ci]
                if gap<self.thickness:
                    wa=p[ci]-self.cr[ci]*n; wb=wa-gap*n
                    self.add_friction(terms,a,0,wa,wb,n,gap,dt)
        for c in self.cfg.get('joints',[]):
            terms.append(term(c['kind'],c['a'],c['b'],ra=c.get('ra',[0,0,0]),rb=c.get('rb',[0,0,0]),
                              radius=c.get('length',0),stiffness=c.get('stiffness',2e6)))
        other=np.array([t.numpy_value() for t in terms],dtype=K.Term.numpy_dtype())
        combined=np.zeros(len(magnetic)+len(other),dtype=K.Term.numpy_dtype())
        combined[:len(magnetic)]=magnetic
        combined[len(magnetic):]=other
        return combined,far,magnetic_count

    def add_friction(self,terms,a,b,wa,wb,n,gap,dt):
        fn=3*self.kappa*(self.thickness-gap)**2
        inva=self.q[a]*np.array([-1,-1,-1,1]); invb=self.q[b]*np.array([-1,-1,-1,1])
        terms.append(term(6,a,b,ra=rotate(inva,wa-self.x[a]),rb=rotate(invb,wb-self.x[b]),
                          normal=n,bias=wa-wb,friction=self.cfg.get('friction',.25)*fn,
                          epsilon=dt*self.cfg.get('friction_velocity',.01)))

    def prepare(self,dt,control):
        self.magnetize()
        self.base_x,self.base_q=self.x.copy(),self.q.copy()
        if control is not None: control(self,self.clock+dt)
        terms,far,mag_count=self.make_terms(dt)
        self.terms=wp.array(terms,dtype=K.Term,device=self.device)
        self.dx,self.dq=self.da(self.base_x,K.V),self.da(self.base_q,K.Q)
        current=np.zeros((self.n,6))
        np.divide(self.v,self.sv,out=current[:,:3],where=self.sv!=0)
        np.divide(self.w,self.sw,out=current[:,3:],where=self.sw!=0)
        predicted=current.copy()
        gravity=np.array(self.cfg.get('gravity',[0,0,-9.81]))
        predicted[:,:3]+=dt*gravity*np.sqrt(self.mass[:,None])*self.free[:,:3]
        self.predicted=self.da(predicted.ravel(),K.D)
        self.J=np.empty((self.n,3,3)); self.inertia_h=np.eye(self.n*6)
        for i in range(self.n):
            R=rotate(np.tile(self.q[i],(3,1)),np.eye(3)).T
            A=np.diag(self.sw[i]); self.J[i]=A@R@np.diag(self.I[i])@R.T@A
            for j in range(3):
                if not self.free[i,3+j]: self.J[i,j,j]=1
            self.inertia_h[6*i:6*i+3,6*i:6*i+3]*=1+dt*self.damp[i,0]
            self.inertia_h[6*i+3:6*i+6,6*i+3:6*i+6]=self.J[i]*(1+dt*self.damp[i,1])
        self.Jw=self.da(self.J,K.M)
        self.far=self.da(np.zeros(self.n*6),K.D)
        if len(far):
            # Taylor-expand the SAME pair energy, avoiding pair-counting mismatch.
            ft=wp.array(far,dtype=K.Term,device=self.device)
            self.u.zero_(); self.E.zero_()
            with wp.Tape() as tape:
                wp.launch(K.potentials,len(far),[self.u,ft,self.da(self.x,K.V),self.da(self.q,K.Q),self.dsv,self.dsw,dt,self.E],device=self.device)
            tape.backward(self.E)
            self.far=self.da(self.u.grad.numpy(),K.D); tape.zero()
        return current.ravel(),mag_count,len(far)

    def evaluate(self,u,dt,gradient=True):
        self.u.assign(np.asarray(u,dtype=np.float64)); self.E.zero_()
        if self.u.grad is not None: self.u.grad.zero_()
        with wp.Tape() as tape:
            wp.launch(K.inertia,self.n,[self.u,self.predicted,self.Jw,self.damping,dt,self.far,self.E],device=self.device)
            if len(self.terms):
                wp.launch(K.potentials,len(self.terms),[self.u,self.terms,self.dx,self.dq,self.dsv,self.dsw,dt,self.E],device=self.device)
        value=float(self.E.numpy()[0])
        g=None
        if gradient:
            tape.backward(self.E)
            g=self.u.grad.numpy().copy()
        tape.zero()
        return value,g

    def hessian(self,u,dt):
        self.u.assign(np.asarray(u,dtype=np.float64));self.H.zero_()
        if len(self.terms):
            wp.launch(K.term_hessians,(len(self.terms),12,12),[self.u,self.terms,self.dx,self.dq,self.dsv,self.dsw,
                      dt,self.cfg.get('hessian_epsilon',2.e-4),self.dactive,self.H],device=self.device)
        return self.inertia_h+self.H.numpy()

    def attempt(self,dt,control):
        u,implicit,explicit=self.prepare(dt,control)
        xi,qi=self.pose_numpy(u,dt)
        if not self.swept_safe(self.x,self.q,xi,qi): u=np.zeros_like(u)
        E,g=self.evaluate(u,dt)
        mass=np.repeat(np.stack([self.mass,self.I.mean(1)],1),3,axis=1).ravel()
        tolerance=self.cfg.get('acceleration_tolerance',.15)
        iterations=0; backtracks=0; residual=np.inf
        start=time.monotonic()
        for iteration in range(self.cfg.get('newton_iterations',35)):
            iterations=iteration+1
            residual=float(np.max(np.abs(g[self.active])/(dt*np.sqrt(mass[self.active]))))
            if residual<tolerance: break
            H=self.hessian(u,dt)[np.ix_(self.active,self.active)]
            values,Q=scipy.linalg.eigh((H+H.T)*.5,check_finite=False,driver='evr')
            projected=np.maximum(values,self.cfg.get('hessian_floor',.03))
            direction=np.zeros_like(u)
            direction[self.active]=-Q@((Q.T@g[self.active])/projected)
            slope=float(g@direction)
            alpha=1.
            xp,qp=self.pose_numpy(u,dt)
            accepted=False
            for ls in range(28):
                candidate=u+alpha*direction
                xn,qn=self.pose_numpy(candidate,dt)
                if self.swept_safe(xp,qp,xn,qn):
                    En,gn=self.evaluate(candidate,dt)
                    if np.isfinite(En) and En<=E+1.e-4*alpha*slope+1.e-12:
                        u,E,g=candidate,En,gn;accepted=True;break
                alpha*=.5;backtracks+=1
            if not accepted:
                return None,{'reason':'line_search','residual':residual,
                             'min_gap':float(np.min(self.gaps(xp,qp)))}
        else:
            xx,qq=self.pose_numpy(u,dt)
            return None,{'reason':'newton_limit','residual':residual,
                         'min_gap':float(np.min(self.gaps(xx,qq)))}
        xn,qn=self.pose_numpy(u,dt)
        cp=self.centers(xn,qn)
        gap=np.linalg.norm(cp[self.ca]-cp[self.cb],axis=1)-self.cr[self.ca]-self.cr[self.cb]
        if np.any((gap<self.thickness)&~self.contact_candidates):
            return None,{'reason':'expand_contact_candidates','residual':residual}
        if not self.swept_safe(self.x,self.q,xn,qn):
            return None,{'reason':'swept_contact','residual':residual}
        self.u.assign(u)
        nx,nq=self.da(self.x,K.V),self.da(self.q,K.Q)
        nv,nw=wp.zeros_like(nx),wp.zeros_like(nx)
        wp.launch(K.integrate,self.n,[self.u,self.dx,self.dq,self.dsv,self.dsw,dt,nx,nq,nv,nw],device=self.device)
        return (nx.numpy(),nq.numpy(),nv.numpy(),nw.numpy()),dict(dt=dt,newton=iterations,backtracks=backtracks,
            residual=residual,min_gap=float(np.min(self.gaps(xn,qn))),implicit_pairs=implicit,explicit_pairs=explicit,
            terms=len(self.terms),energy=E,seconds=time.monotonic()-start,kappa=self.kappa)

    def step(self,dt,control=None,depth=0):
        saved_m,saved_perm=self.m.copy(),self.permanent.copy()
        result,info=self.attempt(dt,control)
        if result is None:
            self.m,self.permanent=saved_m,saved_perm
            if depth>=9: raise RuntimeError(f'Cannot accept step at {self.clock}: {info}')
            # A failed finite-difference Newton solve need not mean the contact
            # penalty is weak. Only stiffen a contact that is nearly exhausted;
            # otherwise refine time without injecting penalty energy at rest.
            if info.get('min_gap',self.thickness)<.10*self.thickness:
                self.kappa=min(self.kappa*2,1.e13)
            print({'retry_time':self.clock,'retry_dt':dt,'depth':depth,
                   'kappa':self.kappa,**info},flush=True)
            self.step(dt/2,control,depth+1);self.step(dt/2,control,depth+1)
            return
        self.x,self.q,self.v,self.w=result
        self.clock+=dt
        info.update(time=self.clock,retry_depth=depth)
        self.history.append(info)
