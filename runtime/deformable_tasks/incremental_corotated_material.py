"""CPU reference material evaluated from unrounded rest-relative increments.

World-coordinate publication can lose the strain direction immediately after
impact. Build F-I directly and evaluate its polar strain in extended precision.
The constitutive law and gamma are unchanged. This is not a GPU implementation.
"""
import numpy as np
from .corotated_material_reference import CorotatedMaterial, cofactor


class IncrementalCorotatedMaterial(CorotatedMaterial):
    def __init__(self,model,position,increment):
        self.__dict__.update(model.__dict__)
        x,delta = np.asarray(position,float),np.asarray(increment,float)
        if x.shape != self.rest.shape or delta.shape != x.shape or not np.isfinite(x+delta).all():
            raise ValueError('Finite material positions/increments required')
        if np.finfo(np.longdouble).eps>=np.finfo(float).eps:
            raise RuntimeError('This CPU reference requires extended precision')
        wide = np.longdouble
        # Rest-relative differences retain rigid translation cancellation before
        # forming the near-identity deformation gradient.
        displacement = x.astype(wide)-self.rest.astype(wide)+delta.astype(wide)
        local = displacement[self.tets]
        E = (local[:,1:]-local[:,:1]).transpose(0,2,1) @ self.Q.astype(wide)
        identity = np.eye(3,dtype=wide)
        F = identity+E
        det = lambda a: np.sum(a[..., :,0]*np.cross(a[..., :,1],a[..., :,2]),axis=-1)
        J = det(F)
        if np.any(J<=0):
            raise ValueError('Positive material volumes required')
        R = F.copy()
        for _ in range(40):
            determinant = det(R)
            inverse_transpose = cofactor(R)/determinant[:,None,None]
            new = .5*(R+inverse_transpose)
            if np.max(abs(new-R))<8*np.finfo(wide).eps:
                R = new
                break
            R = new
        else:
            raise RuntimeError('Extended polar decomposition failed')
        strain = E-(R-identity)
        stretch = R.transpose(0,2,1) @ F
        values,V = np.linalg.eigh(np.asarray(.5*(stretch+stretch.transpose(0,2,1)),float))
        if np.any(values<=0):
            raise ValueError('Positive polar stretch required')
        self._expected = x+delta
        self._strain = np.asarray(strain,float)
        self._cofactor = np.asarray(cofactor(F),float)
        self._volume_strain = np.asarray(J-1,float)
        self._kinematics = (np.asarray(F,float),np.asarray(J,float),np.asarray(R,float),values,V.transpose(0,2,1))
        self._density = np.asarray(self.mu*np.sum(strain*strain,axis=(1,2))+.5*self.lam*(J-1)**2,float)

    def kinematics(self,x):
        if not np.array_equal(np.asarray(x,float),self._expected):
            raise ValueError('Incremental material evaluated at a different stage')
        return self._kinematics

    def strain_at(self,x):
        self.kinematics(x)
        return self._strain

    def energy_gradient(self,x):
        self.kinematics(x)
        P = 2*self.mu*self._strain+self.lam*self._volume_strain[:,None,None]*self._cofactor
        return float(self.volume @ self._density),self.scatter(self.volume[:,None,None]*self.local_gradient(P))

    def directional_damping(self,x,velocity,gamma):
        self.kinematics(x)
        if not np.isfinite(gamma) or gamma<0:
            raise ValueError('Invalid damping coefficient')
        norm = np.sqrt(np.sum(self._strain**2,axis=(1,2))+1e-18)
        g0 = self.local_gradient(self._strain/norm[:,None,None])
        g1 = self.local_gradient(self._cofactor)
        v = np.asarray(velocity,float)[self.tets]
        block = g0*np.sum(g0*v,axis=(1,2))[:,None,None]
        if self.lam>0:
            block += g1*np.sum(g1*v,axis=(1,2))[:,None,None]
        return self.scatter((2*self.mu*gamma*self.volume)[:,None,None]*block)
