"""Independent exponential evolution for a fixed planar contact active set.

Shares assembled linear material matrices, not time integration or contact QP.
Constant target only. Validity is conditional on nonnegative reactions and
nonpenetration; sampled admission does not certify between-sample events.
"""
import numpy as np
from scipy.sparse import coo_matrix, diags, bmat, csc_matrix, eye
from scipy.sparse.linalg import expm_multiply


class FrozenContactOracle:
    def __init__(self, model, active, u, v, target):
        self.model = model
        self.active = np.asarray(active, bool)
        self.u0 = np.asarray(u, float).copy()
        self.target = float(target)
        n = len(model.M); nt = len(model.top)
        if self.active.shape != (len(model.B),):
            raise ValueError('Invalid active mask')
        gap = model.g0 + model.B @ self.u0
        if np.max(abs(gap[self.active]), initial=0) > 1e-10:
            raise ValueError('Active contacts must start closed')
        if np.max(abs((model.B @ v)[self.active]), initial=0) > 1e-8:
            raise ValueError('Active contacts require compatible velocity')
        # Each top normal is tied to the dynamic platen; each bottom normal
        # is fixed. All tangential and interior DOFs remain independent.
        top_dof = 3 * model.top[self.active[:nt]] + 2
        bot_dof = 3 * model.bottom[self.active[nt:]] + 2
        keep = np.ones(n, bool); keep[top_dof] = False; keep[bot_dof] = False
        ids = np.flatnonzero(keep); index = np.full(n, -1, int)
        index[ids] = np.arange(len(ids))
        rr = np.r_[ids, top_dof]; cc = np.r_[index[ids], np.full(len(top_dof), index[-1])]
        self.T = coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(n, len(ids))).tocsc()
        self.mass = np.asarray(self.T.T @ model.M).ravel()
        self.K = (self.T.T @ model.Ktotal @ self.T).tocsc()
        self.D = (self.T.T @ model.D @ self.T).tocsc()
        applied = np.zeros(n); applied[-1] = model.k * target
        self.force = np.asarray(self.T.T @ (applied - model.Ktotal @ self.u0)).ravel()
        vr = np.asarray(self.T.T @ (model.M * v)).ravel() / self.mass
        self.initial_velocity_mapping_error = float(abs(self.T @ vr - v).max())
        if self.initial_velocity_mapping_error > 1e-8:
            raise ValueError('Velocity does not belong to fixed-contact space')
        nr = len(ids); self.omega = max(1., float(np.sqrt(np.max(self.K.diagonal() / self.mass))))
        invm = diags(1 / self.mass)
        # Scale displacement block to avoid the artificial ||K/M|| of the
        # unbalanced first-order state. Affine forcing is an augmented scalar.
        self.A = bmat([[None, self.omega * eye(nr), csc_matrix((nr, 1))],
                       [-invm @ self.K / self.omega, -invm @ self.D, csc_matrix((self.force / self.mass)[:, None])],
                       [csc_matrix((1, nr)), csc_matrix((1, nr)), csc_matrix((1, 1))]], format='csc')
        self.z0 = np.r_[np.zeros(nr), vr, 1.]

    def evolve(self, duration, samples):
        if duration <= 0 or samples < 2:
            raise ValueError('Positive duration and at least two samples required')
        z = expm_multiply(self.A, self.z0, start=0., stop=duration, num=samples,
                          traceA=float(self.A.diagonal().sum()))
        nr = len(self.mass); y = z[:, :nr] / self.omega; vr = z[:, nr:2*nr]
        ar = (self.force[None, :] - (self.K @ y.T).T - (self.D @ vr.T).T) / self.mass
        u = self.u0[None, :] + (self.T @ y.T).T
        v = (self.T @ vr.T).T; acceleration = (self.T @ ar.T).T
        m = self.model
        residual = acceleration * m.M + (m.D @ v.T).T + (m.Ktotal @ u.T).T
        residual[:, -1] -= m.k * self.target
        reaction = np.zeros((samples, len(m.B)))
        nt = len(m.top)
        reaction[:, :nt] = -residual[:, 3*m.top+2]
        reaction[:, nt:] = residual[:, 3*m.bottom+2]
        reaction[:, ~self.active] = 0.
        balance = residual - (m.B.T @ reaction.T).T
        gap = m.g0[None, :] + (m.B @ u.T).T
        metrics = dict(max_force_residual_N=float(abs(balance).max()),
                       min_active_reaction_N=float(reaction[:, self.active].min()) if self.active.any() else None,
                       max_penetration_m=float(max(0., -gap.min())),
                       max_active_gap_m=float(abs(gap[:, self.active]).max()) if self.active.any() else 0.,
                       max_active_normal_speed_m_s=float(abs((m.B @ v.T).T[:, self.active]).max()) if self.active.any() else 0.,
                       initial_velocity_mapping_error_m_s=self.initial_velocity_mapping_error)
        metrics['sampled_admissible'] = (metrics['max_force_residual_N'] < 1e-6 and metrics['max_penetration_m'] < 1e-9
            and (metrics['min_active_reaction_N'] is None or metrics['min_active_reaction_N'] >= -1e-8))
        return dict(time_s=np.linspace(0, duration, samples), u=u, v=v, reaction=reaction, metrics=metrics)
