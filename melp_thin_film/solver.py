"""GPU MELP transfers, surface operators and implicit thin-film dynamics."""
import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator,cg
from . import kernels as K


def fibonacci(n,radius=1.):
    i=np.arange(n);z=1-2*(i+.5)/n;phi=i*np.pi*(3-np.sqrt(5))
    return radius*np.stack([np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z],1)


def initial_state(cfg):
    xs=[];ns=[];ps=[];vs=[];vols=[];soaps=[];hs=[];eg=[];lg=[]
    rng=np.random.default_rng(cfg['seed'])
    for group,b in enumerate(cfg['bubbles']):
        r=b['radius'];center=np.array(b['center']);axes=np.array(b.get('axes',[1.,1.,1.]))
        e=fibonacci(cfg['e_count']);l=fibonacci(cfg['l_count'])
        # A rotated dense material cloud avoids coincident E/L lattices.
        angle=.137;rot=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        l=l@rot.T
        x=center+r*e*axes;p=center+r*l*axes
        n=e/axes;n/=np.linalg.norm(n,axis=1)[:,None]
        ln=l/axes;ln/=np.linalg.norm(ln,axis=1)[:,None]
        az=np.arctan2(l[:,1],l[:,0]);lat=np.arcsin(l[:,2])
        h=cfg['thickness']*(1+.48*np.sin(3*az+2*l[:,2])*np.cos(lat)**2+.22*np.cos(7*lat+2*az)*np.cos(lat))
        area_l=4*np.pi*r*r/cfg['l_count']*np.prod(axes)*np.linalg.norm(l/axes,axis=1)
        gamma=(cfg['sigma0']-cfg['sigma'])/cfg['rt']
        velocity=np.zeros_like(l)
        for k in range(5):
            a=rng.normal(size=3);a/=np.linalg.norm(a)
            strength=cfg.get('vortex_speed',.04)*(1 if k%2==0 else -1)
            velocity+=strength*np.cross(a,l)*np.exp(5*(l@a-1))[:,None]
        velocity+=cfg.get('rotation_speed',.03)*np.cross(np.array([0.,0.,1.]),l)
        velocity-=np.sum(velocity*ln,axis=1)[:,None]*ln
        velocity+=np.array(b.get('velocity',[0.,0.,0.]))
        xs.append(x);ns.append(n);ps.append(p);vs.append(velocity);vols.append(h*area_l);soaps.append(gamma*area_l);hs.append(h)
        eg.extend([group]*len(x));lg.extend([group]*len(p))
    return [np.concatenate(v) for v in [xs,ns,ps,vs,vols,soaps,hs]],np.array(eg,np.int32),np.array(lg,np.int32)


class Solver:
    def __init__(self,cfg,device='cuda:0'):
        wp.init();self.cfg=cfg;self.device=wp.get_device(device);self.time=0.;self.step_count=0
        values,eg,lg=initial_state(cfg);x,n,p,v,vol,soap,lh=values
        self.ne=len(x);self.nl=len(p);self.ng=len(cfg['bubbles']);self.capacity=cfg.get('neighbor_capacity',128)
        self.h=cfg['support']*np.sqrt(4*np.pi*cfg['bubbles'][0]['radius']**2/cfg['e_count'])
        def arr(v,t):return wp.array(v,dtype=t,device=self.device)
        def zeros(size,t):return wp.zeros(size,dtype=t,device=self.device)
        self.x=arr(x,K.V);self.n=arr(n,K.V);self.p=arr(p,K.V);self.v=arr(v,K.V)
        self.vol=arr(vol,K.D);self.soap=arr(soap,K.D);self.lh=arr(lh,K.D)
        self.eg=arr(eg,int);self.lg=arr(lg,int);self.centers=arr([b['center'] for b in cfg['bubbles']],K.V)
        self.other=zeros(self.ne,int);self.directions=zeros(self.ne,K.V);self.gap=zeros(self.ne,K.D);self.share=zeros(self.ne,K.D);self.strength=zeros(self.ne,K.D);self.contact_diag=zeros(self.ne,K.D);self.internal=zeros(self.ne,K.D);self.net_force=zeros(self.ng,K.V);self.normal_tensor=zeros((self.ng,9),K.D)
        self.affine=zeros(self.nl,K.M);self.ln=zeros(self.nl,K.V);self.mean=zeros(self.nl,K.V)
        for name in ['area','curvature','ev','ec','thickness','gamma','gamma_solved','base_t','diag_t','rhs_t','inv_t','diag_n','rhs_n','inv_n','vn','divergence','force','no_share']:
            setattr(self,name,zeros(self.ne,K.D))
        for name in ['newn','density_gradient','momentum','velocity','vt','grad_tmp','geometric_velocity','shifted_x','capillary']:setattr(self,name,zeros(self.ne,K.V))
        self.enb=zeros((self.ne,self.capacity),int);self.lnb=zeros((self.nl,self.capacity),int)
        self.enum=zeros(self.ne,int);self.lnum=zeros(self.nl,int)
        self.cross_nb=zeros((self.ne,self.capacity),int);self.cross_count=zeros(self.ne,int);self.cross_weights=zeros((self.ne,self.capacity),K.D)
        for name in ['lap','coeff_n']:setattr(self,name,zeros((self.ne,self.capacity),K.D))
        self.gradient=zeros((self.ne,self.capacity),K.V);self.weights=zeros((self.nl,self.capacity),K.D)
        for name in ['volume','surface','rank']:setattr(self,name,zeros(self.ng,K.D))
        self.projection=zeros(self.nl,K.D);self.xf=zeros(self.ne,wp.vec3);self.grid=wp.HashGrid(64,64,64,device=self.device)
        self.A_t=LinearOperator((self.ne,self.ne),K.D,self.device,self.t_matvec)
        self.A_n=LinearOperator((self.ne,self.ne),K.D,self.device,self.n_matvec)
        self.rebuild();self.geometry()
        for _ in range(cfg.get('initial_relaxation_steps',0)):self.redistribute()
        self.transfer();self.operators();self.update_volume();x=self.x.numpy()
        current=self.volume.numpy()
        # Equilibrate the discrete sphere's mean curvature, keeping local deviations.
        lap=self.lap.numpy();nb=self.enb.numpy();counts=self.enum.numpy();areas=self.area.numpy();normals=self.n.numpy()
        cap=np.zeros(self.ne)
        for i in range(self.ne):
            ids=nb[i,:counts[i]];cap[i]=-np.sum(lap[i,:counts[i],None]*(x[ids]-x[i])*normals[i])
        pressure=np.bincount(eg,weights=2*cfg['sigma']*cap)/np.bincount(eg,weights=areas)
        self.gas=arr((cfg['atmosphere']+pressure)*current,K.D)
        self.initial_volume=current.copy();self.initial_area=self.surface.numpy().copy()
        self.initial_liquid=float(vol.sum());self.initial_soap=float(soap.sum());self.last_solve={}
        self.check_neighbors()

    def launch(self,fn,n,*args):wp.launch(fn,dim=n,inputs=list(args),device=self.device)

    def rebuild(self):
        self.launch(K.float_positions,self.ne,self.x,self.xf)
        self.grid.build(self.xf,self.h)
        self.launch(K.find_neighbors,self.ne,self.grid.id,self.x,self.eg,self.x,self.eg,self.h,self.enb,self.enum,self.capacity)
        self.launch(K.find_neighbors,self.nl,self.grid.id,self.x,self.eg,self.p,self.lg,self.h,self.lnb,self.lnum,self.capacity)
        self.check_neighbors()

    def check_neighbors(self):
        e=self.enum.numpy();l=self.lnum.numpy()
        if min(e.min(),l.min())<8 or max(e.max(),l.max())>self.capacity:
            raise RuntimeError(f'Invalid support: E {e.min()}..{e.max()}, L {l.min()}..{l.max()}')

    def geometry(self):
        self.launch(K.surface_geometry,self.ne,self.x,self.n,self.enb,self.enum,self.h,self.area,self.newn,self.curvature,self.density_gradient)
        self.n,self.newn=self.newn,self.n

    def transfer(self):
        self.launch(K.interpolate_weights,self.nl,self.x,self.n,self.p,self.lnb,self.lnum,self.h,self.weights,self.ln,self.mean)
        self.ev.zero_();self.ec.zero_();self.momentum.zero_()
        self.launch(K.scatter,self.nl,self.x,self.p,self.v,self.affine,self.ln,self.mean,self.vol,self.soap,self.lnb,self.lnum,self.weights,self.ev,self.ec,self.momentum)
        self.launch(K.collect,self.ne,self.ev,self.ec,self.momentum,self.area,self.thickness,self.gamma,self.velocity)
        if self.cfg.get('geometric_normal',False):self.launch(K.use_geometric_normal,self.ne,self.velocity,self.n,self.geometric_velocity)

    def operators(self):
        self.launch(K.graph,self.ne,self.x,self.n,self.area,self.enb,self.enum,self.h,self.lap,self.gradient)

    def update_volume(self):
        self.volume.zero_();self.surface.zero_()
        self.launch(K.statistics,self.ne,self.x,self.n,self.area,self.eg,self.centers,self.volume,self.surface)

    def redistribute(self):
        self.launch(K.redistribute_surface,self.ne,self.x,self.n,self.enb,self.enum,self.density_gradient,self.other,self.share,
                    float(self.cfg.get('contact_redistribution',False)),self.h,self.cfg.get('redistribute',.001),self.shifted_x)
        self.x,self.shifted_x=self.shifted_x,self.x
        self.rebuild();self.geometry()

    def t_matvec(self,x,y,z,alpha,beta):
        self.launch(K.compatible_gradient,self.ne,x,self.area,self.thickness,self.enb,self.enum,self.gradient,self.cfg['rt'],self.cfg['rho'],self.grad_tmp)
        self.launch(K.compatible_base,self.ne,self.base_t,x,y,z,alpha,beta)
        self.launch(K.compatible_adjoint,self.ne,self.grad_tmp,self.enb,self.enum,self.gradient,z,self.dt,alpha)

    def n_matvec(self,x,y,z,alpha,beta):
        self.rank.zero_();self.launch(K.rank_reduction,self.ne,self.area,self.eg,x,self.rank)
        self.launch(K.normal_matvec,self.ne,self.enb,self.enum,self.coeff_n,self.diag_n,self.area,self.eg,self.gas,self.volume,self.rank,self.dt,x,y,z,alpha,beta)
        if self.ng>1:
            if self.cfg.get('continuous_contact',False):
                self.launch(K.continuous_contact_matvec,self.ne,self.other,self.cross_nb,self.cross_count,self.cross_weights,self.directions,self.n,self.strength,x,z,self.dt,alpha)
            else:self.launch(K.contact_matvec,self.ne,self.other,self.directions,self.n,self.strength,x,z,self.dt,alpha)

    def step(self,dt):
        cfg=self.cfg;self.dt=dt
        self.rebuild();self.geometry();self.transfer();self.operators();self.update_volume()
        # Smooth the predictor; the subsequent implicit solve then enforces gas
        # volume and contact constraints on the final velocity, without undoing
        # those constraints through a post-solve filter.
        for _ in range(cfg.get('normal_smoothing_passes',0)):
            self.launch(K.smooth_normal,self.ne,self.x,self.n,self.area,self.velocity,self.enb,self.enum,self.h,cfg.get('normal_smoothing',.5),self.vn)
            self.launch(K.replace_normal,self.ne,self.velocity,self.n,self.vn)
        gravity=K.V(*cfg.get('gravity',[0.,0.,0.]))
        self.launch(K.compatible_init,self.ne,self.gamma,self.area,self.base_t,self.diag_t,self.rhs_t)
        self.launch(K.compatible_rhs,self.ne,self.velocity,self.n,self.area,self.thickness,self.enb,self.enum,self.gradient,gravity,dt,cfg['rt'],cfg['rho'],cfg.get('gravity_scale',1.),self.diag_t,self.rhs_t)
        self.launch(K.invert_diagonal,self.ne,self.diag_t,self.inv_t)
        wp.copy(self.gamma_solved,self.gamma)
        it_t,err_t,tol_t=cg(self.A_t,self.rhs_t,self.gamma_solved,M=self.inv_t,tol=1.e-7,atol=1.e-14,maxiter=200,check_every=10)
        self.launch(K.update_tangential,self.ne,self.velocity,self.n,self.gamma_solved,self.thickness,self.enb,self.enum,self.gradient,
                    gravity,dt,cfg['rt'],cfg['rho'],cfg.get('gravity_scale',1.),cfg.get('tangent_damping',.02),self.vt)
        if self.ng>1:
            if cfg.get('continuous_contact',False):
                self.launch(K.continuous_contacts,self.ne,self.grid.id,self.x,self.n,self.eg,self.h,self.capacity,self.other,self.directions,self.gap,self.share,self.cross_nb,self.cross_count,self.cross_weights)
            else:self.launch(K.cross_contacts,self.ne,self.grid.id,self.x,self.n,self.eg,self.h,self.other,self.directions,self.gap,self.share)
        self.launch(K.drive_pressure,self.ne,self.x,self.n,self.eg,self.centers,self.time,cfg.get('pressure_amplitude',0.),self.force)
        share=self.share
        if cfg.get('multi_region',False) or not cfg.get('tension_sharing',True):share=self.no_share
        self.launch(K.normal_rhs,self.ne,self.x,self.n,self.velocity,self.area,self.ev,self.gamma_solved,self.enb,self.enum,self.lap,self.eg,
                    self.volume,self.gas,dt,cfg['rho'],cfg['rt'],cfg['sigma0'],cfg['atmosphere'],cfg.get('normal_damping',25.),cfg.get('normal_added_mass',0.),
                    self.internal,self.force,share,cfg.get('preserve_contact_traction',0.),self.diag_n,self.rhs_n,self.coeff_n,self.inv_n)
        if cfg.get('balance_normal_force',True):
            self.net_force.zero_();self.normal_tensor.zero_()
            self.launch(K.force_balance_reduce,self.ne,self.n,self.area,self.eg,self.internal,self.net_force,self.normal_tensor)
            self.launch(K.force_balance_apply,self.ne,self.n,self.area,self.eg,self.net_force,self.normal_tensor,dt,self.rhs_n)
        if cfg.get('multi_region',False):
            self.launch(K.capillary_field,self.ne,self.x,self.area,self.gamma_solved,self.enb,self.enum,self.lap,cfg['sigma0'],cfg['rt'],self.capillary)
            self.launch(K.multi_region_traction,self.ne,self.grid.id,self.x,self.n,self.area,self.gamma_solved,self.capillary,
                        self.other,self.share,self.cross_nb,self.cross_count,self.cross_weights,int(cfg.get('continuous_contact',False)),self.eg,self.gas,self.volume,cfg['sigma0'],cfg['rt'],cfg['atmosphere'],self.h,dt,self.rhs_n)
        if self.ng>1:
            self.contact_diag.zero_()
            if cfg.get('continuous_contact',False):
                self.launch(K.continuous_contact_rhs,self.ne,self.other,self.cross_nb,self.cross_count,self.cross_weights,self.directions,self.gap,self.share,self.n,self.area,dt,cfg.get('contact_stiffness',3.e4),cfg.get('contact_separation',.00015),int(cfg.get('contact_adhesion',True)),self.contact_diag,self.rhs_n,self.strength)
                self.launch(K.contact_preconditioner,self.ne,self.contact_diag,self.inv_n)
            else:self.launch(K.contact_rhs,self.ne,self.other,self.directions,self.gap,self.share,self.n,self.area,dt,cfg.get('contact_stiffness',3.e4),cfg.get('contact_separation',.00015),self.contact_diag,self.rhs_n,self.strength)
        self.vn.zero_()
        it_n,err_n,tol_n=cg(self.A_n,self.rhs_n,self.vn,M=self.inv_n,tol=1.e-7,atol=1.e-14,maxiter=300,check_every=10)
        if not np.isfinite([err_t,err_n]).all() or err_t>max(tol_t*2,1.e-13) or err_n>max(tol_n*2,1.e-13):
            raise RuntimeError(f'Linear solve failed: tangential {it_t} {err_t} {tol_t}; normal {it_n} {err_n} {tol_n}')
        self.launch(K.combine,self.ne,self.vt,self.vn,self.n,self.velocity)
        self.launch(K.remember_normal,self.ne,self.vn,self.n,self.geometric_velocity)
        self.launch(K.dye_thickness,self.ne,self.x,self.n,self.velocity,self.p,self.v,self.lnb,self.lnum,self.weights,self.gradient,self.enb,self.enum,self.divergence)
        self.launch(K.advect_thickness,self.nl,self.lnb,self.lnum,self.weights,self.divergence,self.lh,dt)
        self.launch(K.e_to_l,self.nl,self.x,self.p,self.ln,self.velocity,self.lnb,self.lnum,self.weights,self.mean,self.v,self.affine,cfg.get('affine_strength',.95))
        redistribution=cfg.get('redistribution_iterations',0)
        self.launch(K.advance_e,self.ne,self.x,self.n,self.vn,self.density_gradient,dt,self.h,0. if redistribution else cfg.get('redistribute',.0002))
        if redistribution:
            self.rebuild();self.geometry()
            for _ in range(redistribution):self.redistribute()
        self.launch(K.advance_l,self.nl,self.p,self.v,dt)
        self.rebuild()
        self.launch(K.project_l,self.nl,self.x,self.n,self.p,self.lnb,self.lnum,self.h,self.ln,self.projection)
        self.time+=dt;self.step_count+=1
        self.last_solve=dict(tangential_iterations=it_t,tangential_residual=err_t,tangential_tolerance=tol_t,
                             normal_iterations=it_n,normal_residual=err_n,normal_tolerance=tol_n)

    def state(self):
        return {name:getattr(self,name).numpy() for name in ['x','n','p','v','vol','soap','lh','area','thickness','gamma','eg','lg','share']}

    def refresh(self):
        """Synchronize reconstructed fields with the recorded end-of-step geometry."""
        self.rebuild();self.geometry();self.transfer();self.update_volume()

    def diagnostics(self):
        s=self.state();v=s['v'];h=s['thickness'];gam=s['gamma']
        assert all(np.isfinite(a).all() for a in s.values()),'Nonfinite state'
        assert np.min(h)>0 and np.min(s['lh'])>0 and np.min(gam)>0,'Nonpositive film state'
        assert np.max(gam)<self.cfg['sigma0']/self.cfg['rt'],'Negative surface tension'
        transfer_v=float(self.ev.numpy().sum());transfer_c=float(self.ec.numpy().sum())
        assert abs(transfer_v/self.initial_liquid-1)<1e-10
        assert abs(transfer_c/self.initial_soap-1)<1e-10
        ratios=[];centroids=[]
        for g in range(self.ng):
            points=s['x'][s['eg']==g];eigen=np.linalg.eigvalsh(np.cov(points.T))
            ratios.append(float(np.sqrt(eigen[-1]/eigen[0])));centroids.append(points.mean(0).tolist())
        return dict(time=self.time,step=self.step_count,covariance_axis_ratios=ratios,geometric_centroids=centroids,liquid_transfer_relative_error=abs(transfer_v/self.initial_liquid-1),
                    soap_transfer_relative_error=abs(transfer_c/self.initial_soap-1),thickness_min=float(h.min()),thickness_max=float(h.max()),
                    material_thickness_min=float(s['lh'].min()),material_thickness_max=float(s['lh'].max()),max_speed=float(np.linalg.norm(v,axis=1).max()),
                    enclosed_volume=self.volume.numpy().tolist(),area=self.surface.numpy().tolist(),initial_volume=self.initial_volume.tolist(),
                    max_projection=float(self.projection.numpy().max()),contact_points=int(np.sum(self.share.numpy()>.5)),min_signed_gap=float(-self.gap.numpy().max()),**self.last_solve)
