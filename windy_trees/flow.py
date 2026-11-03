"""Smooth pathlines from recorded SPH particle positions and velocities."""
from functools import lru_cache
from pathlib import Path
import numpy as np

def speed_color(speed):
    """Cool slow air, pale neutral midrange, warm fast air; speeds are m/s."""
    knots=np.array([0.,5.,10.,16.])
    palette=np.array([[.13,.34,.46],[.29,.60,.66],[.63,.78,.72],[.96,.69,.31]])
    speed=np.asarray(speed)
    return np.stack([np.interp(speed,knots,palette[:,k]) for k in range(3)],axis=-1)

def smoothstep(x):
    x=np.clip(x,0.,1.)
    return x*x*(3-2*x)

class Pathlines:
    """Interpolate measured particle paths; never join a recycled boundary crossing."""
    def __init__(self,states,fps,history=6,subdivisions=3,max_particles=400):
        self.states=Path(states);self.fps=fps;self.history=history;self.subdivisions=subdivisions
        self.max_particles=max_particles

    @lru_cache(maxsize=40)
    def state(self,frame):
        with np.load(self.states/f'{frame:05d}.npz') as data:
            stride=max(1,len(data['air_p'])//self.max_particles) if self.max_particles else 1
            return data['air_p'][::stride].copy(),data['air_v'][::stride].copy()

    def curves(self,frame):
        frames=list(range(max(0,frame-self.history),frame+1))
        if len(frames)<2:return []
        history=[self.state(f) for f in frames]
        positions=np.stack([s[0] for s in history]);velocities=np.stack([s[1] for s in history])
        curves=[];dt=1/self.fps
        for particle in range(positions.shape[1]):
            p=positions[:,particle];v=velocities[:,particle]
            jumps=np.flatnonzero(np.linalg.norm(np.diff(p,axis=0),axis=1)>2.0)
            start=int(jumps[-1]+1) if len(jumps) else 0
            p=p[start:];v=v[start:]
            if len(p)<2:continue
            point=p[-1]
            visibility=(1-smoothstep((abs(point[1])-1.5)/1.2))
            visibility*=smoothstep((point[2]-.12)/.45)*(1-smoothstep((point[2]-5.0)/.9))
            visibility*=1-smoothstep((abs(point[0])-4.0)/1.2)
            if visibility<.01:continue
            dense=[];speeds=[]
            for j in range(len(p)-1):
                for t in np.linspace(0,1,self.subdivisions,endpoint=False):
                    # Cubic Hermite interpolation uses the measured endpoint velocities.
                    dense.append((2*t**3-3*t*t+1)*p[j]+(t**3-2*t*t+t)*dt*v[j]+
                                 (-2*t**3+3*t*t)*p[j+1]+(t**3-t*t)*dt*v[j+1])
                    speeds.append(np.linalg.norm((1-t)*v[j]+t*v[j+1]))
            dense.append(p[-1]);speeds.append(np.linalg.norm(v[-1]))
            dense=np.asarray(dense);speed=np.asarray(speeds)
            # Shorter traces arise naturally from slower particles over the same time interval.
            if np.linalg.norm(dense[-1]-dense[0])<.035:continue
            age=np.linspace(0,1,len(dense));fade=smoothstep(age)*visibility
            curves.append(dict(particle=particle,points=dense,speed=speed,fade=fade))
        return curves

    def mesh(self,frame,width=.006):
        vertices=[];faces=[];colors=[];base=0;sides=6
        for curve in self.curves(frame):
            points=curve['points'];tangent=np.gradient(points,axis=0)
            tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1.e-9)
            ref=np.tile([0.,0.,1.],(len(points),1));ref[abs(tangent[:,2])>.90]=[0,1,0]
            u=np.cross(tangent,ref);u/=np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1.e-9)
            v=np.cross(tangent,u);angles=np.arange(sides)*2*np.pi/sides
            radius=width*(.15+.85*np.sqrt(curve['fade']))
            ring=points[:,None,:]+radius[:,None,None]*(np.cos(angles)[None,:,None]*u[:,None,:]+np.sin(angles)[None,:,None]*v[:,None,:])
            vertices.extend(ring.reshape(-1,3).tolist())
            color=np.concatenate([speed_color(curve['speed']),(.36*curve['fade'])[:,None]],axis=1)
            colors.extend(np.repeat(color,sides,axis=0).tolist())
            for j in range(len(points)-1):
                for k in range(sides):
                    nxt=(k+1)%sides
                    faces.append([base+j*sides+k,base+j*sides+nxt,base+(j+1)*sides+nxt,base+(j+1)*sides+k])
            base+=len(points)*sides
        return vertices,faces,np.asarray(colors,dtype=np.float32).reshape(-1,4)


class WindStreaks(Pathlines):
    """Dense, neutral wind filaments from measured SPH trajectories."""
    def __init__(self,states,fps,history=24,subdivisions=2):
        super().__init__(states,fps,history,subdivisions,max_particles=0)

    def visibility(self,points):
        x,y,z=points[...,0],points[...,1],points[...,2]
        fade=smoothstep((x+4.72)/.18)*(1-smoothstep((x-4.3)/1.3))
        fade*=1-smoothstep((abs(y)-1.7)/1.)
        fade*=smoothstep((z-.2)/.35)*(1-smoothstep((z-4.8)/.7))
        return fade

    def curves(self,frame):
        frames=list(range(max(0,frame-self.history),frame+1))
        if len(frames)<2:return []
        states=[self.state(f) for f in frames]
        p=np.stack([s[0] for s in states]);v=np.stack([s[1] for s in states]);dt=1/self.fps
        candidates=np.arange(0,p.shape[1],2)
        keep=candidates[np.any(self.visibility(p[:,candidates])>.015,axis=0)]
        if not len(keep):return []
        p=p[:,keep];v=v[:,keep];count=len(p)
        jump=np.linalg.norm(np.diff(p,axis=0),axis=2)>2.
        t=np.linspace(0,1,self.subdivisions,endpoint=False)[None,:,None,None]
        points=((2*t**3-3*t*t+1)*p[:-1,None]+(t**3-2*t*t+t)*dt*v[:-1,None]+
                (-2*t**3+3*t*t)*p[1:,None]+(t**3-t*t)*dt*v[1:,None])
        speed=np.linalg.norm((1-t)*v[:-1,None]+t*v[1:,None],axis=3)
        points=np.concatenate([points.reshape(-1,len(keep),3),p[-1:]],axis=0)
        speed=np.concatenate([speed.reshape(-1,len(keep)),np.linalg.norm(v[-1:],axis=2)],axis=0)
        result=[]
        for j,particle in enumerate(keep):
            cuts=np.flatnonzero(jump[:,j]);starts=np.r_[0,cuts+1];ends=np.r_[cuts,count-1]
            for a,b in zip(starts,ends):
                start=int(a)*self.subdivisions;end=int(b)*self.subdivisions+1
                line=points[start:end,j]
                if len(line)<3:continue
                age=np.arange(start,end)/max(1,len(points)-1)
                local=np.linspace(0,1,len(line))
                fade=smoothstep(age/.25)*smoothstep(local/.06)*smoothstep((1-local)/.025)
                fade*=self.visibility(line)
                if fade.max()<.015:continue
                result.append(dict(particle=int(particle),points=line,speed=speed[start:end,j],fade=fade,
                                   first_frame=frames[int(a)],last_frame=frames[int(b)]))
        return result

    def mesh(self,frame,width=.0020):
        vertices=[];faces=[];colors=[];base=0;sides=4
        for curve in self.curves(frame):
            points=curve['points'];tangent=np.gradient(points,axis=0)
            tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1.e-9)
            ref=np.tile([0.,0.,1.],(len(points),1));ref[abs(tangent[:,2])>.90]=[0,1,0]
            u=np.cross(tangent,ref);u/=np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1.e-9)
            v=np.cross(tangent,u);angle=np.arange(sides)*2*np.pi/sides
            ring=points[:,None,:]+width*(np.cos(angle)[None,:,None]*u[:,None,:]+np.sin(angle)[None,:,None]*v[:,None,:])
            vertices.append(ring.reshape(-1,3))
            color=np.ones((len(points),4),np.float32);color[:,:3]=[.62,.66,.66]
            color[:,3]=.12*curve['fade'];colors.append(np.repeat(color,sides,axis=0))
            j=np.arange(len(points)-1)[:,None]*sides;k=np.arange(sides)[None,:]
            faces.append(np.stack(np.broadcast_arrays(base+j+k,base+j+(k+1)%sides,
                base+j+sides+(k+1)%sides,base+j+sides+k),axis=-1).reshape(-1,4))
            base+=len(points)*sides
        if not vertices:return [],[],np.empty((0,4),np.float32)
        return np.concatenate(vertices).tolist(),np.concatenate(faces).tolist(),np.concatenate(colors)
