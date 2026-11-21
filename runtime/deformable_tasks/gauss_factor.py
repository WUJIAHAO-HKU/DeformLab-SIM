"""Exact block reduction of the real Gauss [2/2] denominator.

P=I-h*A/2+h^2*A^2/12=(I-a*A)(I-conj(a)*A),
a=h*(1/4+i*sqrt(3)/12). For each first-order elastodynamic resolvent,
eliminate displacement and solve (M+a*D+a^2*K)*v. Only one complex reduced
factorization is stored; its conjugate is applied by conjugating the RHS.
This changes linear algebra only, not the integrator or contact equations.
"""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu


class GaussReducedFactor:
    def __init__(self,oracle,dt):
        self.oracle=oracle;self.dt=dt;self.a=dt*(.25+1j*np.sqrt(3)/12)
        self.n=len(oracle.mass)
        self.factor=splu((diags(oracle.mass)+self.a*oracle.D+self.a*self.a*oracle.K).tocsc())
        self.last_residual=0.;self.last_imaginary_residual=0.

    def resolvent(self,rhs,conjugate=False):
        o=self.oracle;n=self.n;a=np.conj(self.a) if conjugate else self.a
        by=rhs[:n];bv=rhs[n:2*n];bc=rhs[-1]
        forcing=o.mass*bv-a*(o.K@by)/o.omega+a*o.force*bc
        velocity=np.conj(self.factor.solve(np.conj(forcing))) if conjugate else self.factor.solve(forcing)
        return np.r_[by+a*o.omega*velocity,velocity,bc]

    def solve(self,rhs):
        rhs=np.asarray(rhs)
        if rhs.shape!=(2*self.n+1,) or np.iscomplexobj(rhs) or not np.isfinite(rhs).all():
            raise ValueError('Gauss factor requires a finite real state vector')
        result=self.resolvent(self.resolvent(rhs.astype(complex)),conjugate=True)
        self.last_imaginary_residual=float(abs(result.imag).max())
        answer=result.real
        applied=self.oracle.A@answer
        residual=answer-.5*self.dt*applied+self.dt**2/12*(self.oracle.A@applied)-rhs
        self.last_residual=float(abs(residual).max())
        if self.last_imaginary_residual>1e-11*max(1.,float(abs(answer).max())) or self.last_residual>1e-9*max(1.,float(abs(rhs).max())):
            raise RuntimeError('Reduced Gauss factor failed algebraic residual admission')
        return answer
