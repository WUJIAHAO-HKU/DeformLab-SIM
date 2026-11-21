"""Read saved DOP853 dense polynomial coefficients without reintegrating an ODE."""
import numpy as np


class DensePlateReference:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as data:
            self.t=np.array(data['t_old']);self.h=np.array(data['h'])
            self.y=np.array(data['y_old']);self.F=np.array(data['F'])
        n=len(self.t)
        if (not n or self.t.shape!=(n,) or self.h.shape!=(n,) or self.y.ndim!=2 or
            self.y.shape[0]!=n or self.F.shape!=(n,7,self.y.shape[1]) or np.any(self.h<=0)
            or np.any(np.diff(self.t)<=0) or not all(np.isfinite(a).all() for a in (self.t,self.h,self.y,self.F))):
            raise ValueError('Invalid dense reference coefficients')
        if not np.allclose(self.t[1:],(self.t+self.h)[:-1],rtol=0,atol=1e-15):
            raise ValueError('Noncontiguous dense reference')

    def __call__(self,time):
        time=float(time)
        if not np.isfinite(time) or time<self.t[0] or time>self.t[-1]+self.h[-1]:
            raise ValueError('Reference evaluation outside saved interval')
        i=max(0,min(len(self.t)-1,int(np.searchsorted(self.t,time,side='left')-1)))
        z=(time-self.t[i])/self.h[i];result=np.zeros_like(self.y[i])
        for j,f in enumerate(self.F[i,::-1]):
            result+=f;result*=z if j%2==0 else 1-z
        return result+self.y[i]
