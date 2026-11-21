"""Cached analytic endpoint Jacobian for the objective two-point material force.

The derivative holds the first endpoint fixed. It is generally nonsymmetric.
The thresholded tiny-increment branch is differentiated within its branch;
no differentiability is asserted on the branch boundary.
"""
import numpy as np
from .corotated_material_reference import cofactor
from .corotated_discrete_work import density_difference


class DiscreteMaterialTangent:
    def __init__(self, model, x0, x1):
        self.model = model
        F0, J0, _, s0, Vh0 = model.kinematics(x0)
        F1, J1, R1, s1, Vh1 = model.kinematics(x1)
        self.F1 = F1
        self.Fmid = .5 * (F0 + F1)
        dF = F1 - F0
        self.dC = self.Fmid.transpose(0, 2, 1) @ dF + dF.transpose(0, 2, 1) @ self.Fmid
        Cmid = .5 * (F0.transpose(0, 2, 1) @ F0 + F1.transpose(0, 2, 1) @ F1)
        values, self.V = np.linalg.eigh(Cmid)
        if not np.all(values > 0):
            raise ValueError('Positive metric required')
        self.roots = np.sqrt(values)
        VT = self.V.transpose(0, 2, 1)
        self.Cinv = (self.V / values[:, None, :]) @ VT
        self.Cinvhalf = (self.V / self.roots[:, None, :]) @ VT
        self.Jmid = np.sqrt(np.prod(values, axis=1))
        self.G = model.mu * (np.eye(3) - self.Cinvhalf) + .5 * model.lam * (
            (self.Jmid - 1) * self.Jmid)[:, None, None] * self.Cinv
        difference = density_difference(model,F0,J0,s0,Vh0,F1,J1,s1,Vh1,self.dC)
        self.B = np.sum(self.dC ** 2, axis=(1, 2))
        scale = np.maximum(np.linalg.norm(Cmid, axis=(1, 2)), 1.)
        self.active = self.B > np.finfo(float).eps * scale ** 2
        self.alpha = np.zeros_like(self.B)
        np.divide(difference - np.sum(self.G * self.dC, axis=(1, 2)), self.B,
                  out=self.alpha, where=self.active)
        self.Gbar = self.G + self.alpha[:, None, None] * self.dC
        self.P1 = 2 * model.mu * (F1 - R1) + model.lam * (J1 - 1)[:, None, None] * cofactor(F1)

    def product(self, direction):
        model = self.model
        direction = np.asarray(direction, float)
        if direction.shape != model.rest.shape or not np.isfinite(direction).all():
            raise ValueError('Invalid tangent direction')
        d = direction[model.tets]
        Z = (d[:, 1:] - d[:, :1]).transpose(0, 2, 1) @ model.Q
        dC = Z.transpose(0, 2, 1) @ self.F1 + self.F1.transpose(0, 2, 1) @ Z
        dCmid = .5 * dC
        local = self.V.transpose(0, 2, 1) @ dCmid @ self.V
        droot = self.V @ (local / (self.roots[:, :, None] + self.roots[:, None, :])) @ self.V.transpose(0, 2, 1)
        dinvhalf = -self.Cinvhalf @ droot @ self.Cinvhalf
        dinv = -self.Cinv @ dCmid @ self.Cinv
        dJ = .5 * self.Jmid * np.sum(self.Cinv * dCmid, axis=(1, 2))
        dG = -model.mu * dinvhalf + .5 * model.lam * (
            ((2 * self.Jmid - 1) * dJ)[:, None, None] * self.Cinv +
            ((self.Jmid - 1) * self.Jmid)[:, None, None] * dinv)
        dN = np.sum(self.P1 * Z - dG * self.dC - self.G * dC, axis=(1, 2))
        dB = 2 * np.sum(self.dC * dC, axis=(1, 2))
        dalpha = np.zeros_like(self.B)
        np.divide(dN - self.alpha * dB, self.B, out=dalpha, where=self.active)
        dGbar = dG + dalpha[:, None, None] * self.dC + self.alpha[:, None, None] * dC
        dP = Z @ self.Gbar + 2 * self.Fmid @ dGbar
        return model.scatter(model.volume[:, None, None] * model.local_gradient(dP))
