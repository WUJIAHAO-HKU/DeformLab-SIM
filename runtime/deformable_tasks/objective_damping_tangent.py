"""Analytic directional derivative of the existing objective gamma operator.

This differentiates D(x)v in both position and velocity. It does not substitute
Kelvin-Voigt damping, clip eigenvalues, or change the native gamma definition.
"""
import numpy as np
from .corotated_material_reference import cofactor, cofactor_direction


class ObjectiveDampingTangent:
    def __init__(self, model, position, velocity, gamma):
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError('Finite nonnegative gamma required')
        self.model = model
        self.F, _, self.R, self.s, self.Vh = model.kinematics(position)
        velocity = np.asarray(velocity, float)
        if velocity.shape != model.rest.shape or not np.isfinite(velocity).all():
            raise ValueError('Finite nodal velocity required')
        self.v = velocity[model.tets]
        self.strain = model.strain_at(position) if hasattr(model,'strain_at') else self.F-self.R
        self.norm = np.sqrt(np.sum(self.strain**2, axis=(1, 2))+1e-18)
        self.g0 = model.local_gradient(self.strain/self.norm[:, None, None])
        self.g1 = model.local_gradient(cofactor(self.F))
        self.scale = 2*model.mu*gamma*model.volume

    def diagonal(self):
        block = self.g0**2
        if self.model.lam > 0:
            block = block+self.g1**2
        return self.model.scatter(self.scale[:, None, None]*block)

    def product(self, position_direction, velocity_direction):
        model = self.model
        dx, dv = np.asarray(position_direction, float), np.asarray(velocity_direction, float)
        if any(a.shape != model.rest.shape or not np.isfinite(a).all() for a in (dx, dv)):
            raise ValueError('Finite nodal tangent directions required')
        local = dx[model.tets]
        dF = (local[:, 1:]-local[:, :1]).transpose(0, 2, 1) @ model.Q
        B = self.R.transpose(0, 2, 1) @ dF
        V = self.Vh.transpose(0, 2, 1)
        skew = self.Vh @ (B-B.transpose(0, 2, 1)) @ V
        omega = V @ (skew/(self.s[:, :, None]+self.s[:, None, :])) @ self.Vh
        dstrain = dF-self.R @ omega
        dot = np.sum(self.strain*dstrain, axis=(1, 2))
        dg0 = model.local_gradient(dstrain/self.norm[:, None, None]
                                  -self.strain*(dot/self.norm**3)[:, None, None])
        dg1 = model.local_gradient(cofactor_direction(self.F, dF))
        local_dv = dv[model.tets]
        def variation(g, dg):
            gv = np.sum(g*self.v, axis=(1, 2))
            d_gv = np.sum(dg*self.v+g*local_dv, axis=(1, 2))
            return dg*gv[:, None, None]+g*d_gv[:, None, None]
        result = variation(self.g0, dg0)
        if model.lam > 0:
            result += variation(self.g1, dg1)
        return model.scatter(self.scale[:, None, None]*result)
