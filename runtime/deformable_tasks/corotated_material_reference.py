"""Finite-strain corotated material oracle for force/work matching.

W=mu*||F-R||^2 + lambda/2*(det(F)-1)^2. Matches the material energy
used by the existing captured ARAP/volume diagnostic; NOT a native whole-step
solver or contact model. Positive-volume branch only. No projection/filtering.
"""
import numpy as np


def cofactor(F):
    return np.stack((np.cross(F[..., :,1],F[..., :,2]),np.cross(F[..., :,2],F[..., :,0]),np.cross(F[..., :,0],F[..., :,1])),axis=-1)


def cofactor_direction(F,dF):
    return np.stack((np.cross(dF[..., :,1],F[..., :,2])+np.cross(F[..., :,1],dF[..., :,2]),
        np.cross(dF[..., :,2],F[..., :,0])+np.cross(F[..., :,2],dF[..., :,0]),
        np.cross(dF[..., :,0],F[..., :,1])+np.cross(F[..., :,0],dF[..., :,1])),axis=-1)


class CorotatedMaterial:
    def __init__(self,rest,tets,young,poisson):
        self.rest=np.asarray(rest,float);self.tets=np.asarray(tets,int)
        if not np.isfinite([young,poisson]).all() or young<=0 or not 0<=poisson<.5:raise ValueError('Invalid elastic material')
        self.mu=young/(2*(1+poisson));self.lam=young*poisson/((1+poisson)*(1-2*poisson))
        r=self.rest[self.tets];Dm=(r[:,1:]-r[:,:1]).transpose(0,2,1)
        self.volume=abs(np.linalg.det(Dm))/6
        if not np.all(self.volume>1e-18):raise ValueError('Degenerate rest tetrahedra')
        self.Q=np.linalg.inv(Dm)

    def kinematics(self,x):
        x=np.asarray(x,float)
        if x.shape!=self.rest.shape or not np.isfinite(x).all():raise ValueError('Invalid current positions')
        p=x[self.tets];F=(p[:,1:]-p[:,:1]).transpose(0,2,1)@self.Q;J=np.linalg.det(F)
        if not np.all(J>0):raise ValueError('Corotated oracle requires positive volumes')
        U,s,Vh=np.linalg.svd(F);R=U@Vh
        return F,J,R,s,Vh

    def local_gradient(self,tensor):
        corners=(tensor@self.Q.transpose(0,2,1)).transpose(0,2,1)
        return np.concatenate((-corners.sum(axis=1,keepdims=True),corners),axis=1)

    def scatter(self,local):
        result=np.zeros_like(self.rest)
        np.add.at(result,self.tets.ravel(),local.reshape(-1,3))
        return result

    def energy_gradient(self,x):
        F,J,R,s,_=self.kinematics(x)
        density=self.mu*np.sum((s-1)**2,axis=1)+.5*self.lam*(J-1)**2
        P=2*self.mu*(F-R)+self.lam*(J-1)[:,None,None]*cofactor(F)
        gradient=self.scatter(self.volume[:,None,None]*self.local_gradient(P))
        return float(self.volume@density),gradient

    def hessian_product(self,x,direction):
        F,J,R,s,Vh=self.kinematics(x);d=np.asarray(direction,float)[self.tets]
        dF=(d[:,1:]-d[:,:1]).transpose(0,2,1)@self.Q
        # Polar differential from a Sylvester equation, well defined also at
        # repeated singular values (e.g. the undeformed isotropic state).
        B=R.transpose(0,2,1)@dF;V=Vh.transpose(0,2,1)
        skew=Vh@(B-B.transpose(0,2,1))@V
        omega=V@(skew/(s[:,:,None]+s[:,None,:]))@Vh
        dR=R@omega;C=cofactor(F);dJ=np.sum(C*dF,axis=(1,2))
        dP=2*self.mu*(dF-dR)+self.lam*(dJ[:,None,None]*C+(J-1)[:,None,None]*cofactor_direction(F,dF))
        return self.scatter(self.volume[:,None,None]*self.local_gradient(dP))

    def directional_damping(self,x,velocity,gamma):
        if not np.isfinite(gamma) or gamma<0:raise ValueError('Invalid damping coefficient')
        F,J,R,_,_=self.kinematics(x);strain=F-R
        norm=np.sqrt(np.sum(strain*strain,axis=(1,2))+1e-18)
        g0=self.local_gradient(strain/norm[:,None,None]);g1=self.local_gradient(cofactor(F))
        v=np.asarray(velocity,float)[self.tets]
        block=g0*np.sum(g0*v,axis=(1,2))[:,None,None]
        if self.lam>0:block+=g1*np.sum(g1*v,axis=(1,2))[:,None,None]
        # This is the existing directional gamma operator, NOT gamma*K.
        return self.scatter((2*self.mu*gamma*self.volume)[:,None,None]*block)
