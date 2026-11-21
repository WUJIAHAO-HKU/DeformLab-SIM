"""Coupled finite-strain, objective-damping and driven-plate Radau reference.

CPU diagnostic with fixed, planar closed contact sets and one translating plate.
No activation/release, native replacement or general robot acceptance is implied.
Both collocation stages evolve geometry, elasticity, damping and drive together.
Material quadrature and angular RK defects remain explicit, not physical losses.
The target moves linearly between the supplied endpoints over this one interval.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from .objective_damping_tangent import ObjectiveDampingTangent
from .incremental_corotated_material import IncrementalCorotatedMaterial

A = np.array([[5/12, -1/12], [3/4, 1/4]])
B = np.array([3/4, 1/4])
C = np.array([1/3, 1.])


def step(model, mass, x, v, q, w, bottom, top, table, target0, target1, h,
         plate_mass=.2, stiffness=1e6, damping=4000., material_gamma=.05,
         linear_damping=.1, tolerance_N=1e-8, max_iterations=30, record_motion=False):
    x, v, mass = np.asarray(x, float), np.asarray(v, float), np.asarray(mass, float)
    bottom, top = np.asarray(bottom, int), np.asarray(top, int)
    if (x.shape != model.rest.shape or v.shape != x.shape or mass.shape != (len(x),)
            or not all(np.isfinite(a).all() for a in (x, v, mass)) or np.any(mass <= 0)
            or not np.isfinite([q,w,table,target0,target1,h,plate_mass,stiffness,damping,
                                material_gamma,linear_damping,tolerance_N]).all()
            or min(h, plate_mass, tolerance_N) <= 0
            or min(stiffness, damping, material_gamma, linear_damping) < 0):
        raise ValueError('Finite states and valid physical parameters required')
    if (len(np.intersect1d(bottom, top)) or any(ids.ndim != 1 or len(np.unique(ids)) != len(ids)
            or np.any(ids < 0) or np.any(ids >= len(x)) for ids in (bottom, top))):
        raise ValueError('Invalid fixed contact sets')
    maxabs = lambda a: float(np.max(abs(a), initial=0.))
    if max(maxabs(x[bottom, 2]-table), maxabs(x[top, 2]-q)) > 1e-12:
        raise ValueError('Initial contacts must be closed')
    if max(maxabs(v[bottom, 2]), maxabs(v[top, 2]-w)) > 1e-10:
        raise ValueError('Initial contact velocities must match')
    model.kinematics(x)
    mask = np.ones(x.size, bool)
    mask[3*bottom+2] = False
    mask[3*top+2] = False
    ids = np.flatnonzero(mask)
    n = len(ids)+1
    m = mass[:, None]
    def expand(z):
        out = np.zeros(x.size)
        out[ids] = z[:-1]
        out[3*top+2] = z[-1]
        return out.reshape(x.shape)
    def reduce(soft, plate):
        return np.r_[soft.ravel()[ids], plate+soft[top, 2].sum()]
    u0 = np.r_[v.ravel()[ids], w]
    targets = target0+C*(target1-target0)
    target_rate = (target1-target0)/h
    def evaluate(acceleration):
        velocities = u0+h*A @ acceleration
        displacements = h*A @ velocities
        rows, stages = [], []
        for i in range(2):
            X, V, acc = x+expand(displacements[i]), expand(velocities[i]), expand(acceleration[i])
            Q, W = q+displacements[i, -1], velocities[i, -1]
            material = IncrementalCorotatedMaterial(model,x,expand(displacements[i]))
            _, elastic = material.energy_gradient(X)
            drag = material.directional_damping(X, V, material_gamma)+linear_damping*m*V
            soft = m*acc+elastic+drag
            drive = stiffness*(Q-targets[i])+damping*W
            rows.append(reduce(soft, plate_mass*acceleration[i, -1]+drive))
            stages.append((X,V,acc,Q,W,elastic,drag,soft,drive,material))
        return np.array(rows), stages, displacements, velocities
    acceleration = np.zeros((2, n))
    history, linear_counts = [], []
    # Block-diagonal approximation only; residual/JVP use the full operators.
    diagD = ObjectiveDampingTangent(model, x, v, material_gamma).diagonal()+linear_damping*m
    redM = np.r_[np.repeat(mass, 3)[ids], plate_mass+mass[top].sum()]
    redD = np.r_[diagD.ravel()[ids], damping+diagD[top, 2].sum()]
    pre_blocks = redM[:, None, None]*np.eye(2)+h*redD[:, None, None]*A
    pre_blocks[-1] += stiffness*h*h*(A @ A)
    inverses = np.linalg.inv(pre_blocks)
    pre = LinearOperator((2*n, 2*n), matvec=lambda z: np.einsum('nij,jn->in', inverses, z.reshape(2,n)).ravel(), dtype=float)
    for iteration in range(max_iterations+1):
        residual, stages, displacements, velocities = evaluate(acceleration)
        error = maxabs(residual)
        history.append(error)
        if error < tolerance_N:
            break
        if iteration == max_iterations:
            raise RuntimeError('Coupled Radau Newton failed: '+repr(history))
        tangents = [ObjectiveDampingTangent(s[9], s[0], s[1], material_gamma) for s in stages]
        def product(flat):
            da = flat.reshape(2, n)
            du = h*A @ da
            dx = h*A @ du
            rows = []
            for i in range(2):
                dX, dV, dacc = expand(dx[i]), expand(du[i]), expand(da[i])
                soft = (m*dacc+stages[i][9].hessian_product(stages[i][0], dX)
                        +tangents[i].product(dX,dV)+linear_damping*m*dV)
                rows.append(reduce(soft, plate_mass*da[i,-1]+stiffness*dx[i,-1]+damping*du[i,-1]))
            return np.asarray(rows).ravel()
        operator = LinearOperator((2*n, 2*n), matvec=product, dtype=float)
        counts = []
        delta, status = gmres(operator, -residual.ravel(), M=pre, rtol=1e-10, atol=1e-12,
                             restart=min(2*n, 100), maxiter=100, callback=counts.append, callback_type='pr_norm')
        linear_counts.append(len(counts))
        if status != 0 or np.linalg.norm(product(delta)+residual.ravel()) > max(1e-11,1e-9*np.linalg.norm(residual)):
            raise RuntimeError('Coupled Radau GMRES failed: '+str(status))
        delta = delta.reshape(2,n)
        alpha = 1.
        for _ in range(30):
            candidate = acceleration+alpha*delta
            try:
                trial = evaluate(candidate)[0]
            except ValueError:
                alpha *= .5
                continue
            if np.linalg.norm(trial) < np.linalg.norm(residual):
                acceleration = candidate
                break
            alpha *= .5
        else:
            raise RuntimeError('Coupled Radau positive-volume line search failed: '+repr(dict(h=h,residual_history_N=history)))
    X, V, acc, Q, W, _, _, _, _, _ = stages[1]
    U0, _ = model.energy_gradient(x)
    U1, _ = model.energy_gradient(X)
    dk = float(.5*np.sum(m*(V*V-v*v))+.5*plate_mass*(W*W-w*w))
    spring = .5*stiffness*((Q-target1)**2-(q-target0)**2)
    elastic_work = h*sum(B[i]*float(np.sum(s[5]*s[1])) for i,s in enumerate(stages))
    material_work = h*sum(B[i]*float(np.sum(s[6]*s[1])) for i,s in enumerate(stages))
    drive_loss = h*damping*float(B @ (velocities[:,-1]**2))
    target_work = -h*stiffness*target_rate*sum(B[i]*(s[3]-targets[i]) for i,s in enumerate(stages))
    lower = np.array([s[7][bottom,2] for s in stages])
    upper = np.array([-s[7][top,2] for s in stages])
    contact_work = 0.
    soft_torque = np.zeros(3)
    external_impulse = np.zeros(3)
    for i,s in enumerate(stages):
        contact = np.zeros_like(x)
        contact[bottom,2] = lower[i]
        contact[top,2] = -upper[i]
        contact_work += h*B[i]*float(np.sum(contact*s[1])+upper[i].sum()*s[4])
        soft_torque += h*B[i]*np.cross(s[0], contact-linear_damping*m*s[1]).sum(axis=0)
        external_impulse += h*B[i]*(np.array([0.,0.,lower[i].sum()-s[8]])-linear_damping*np.sum(m*s[1],axis=0))
    da_soft = stages[0][2]-stages[1][2]
    kinetic_loss = h*h/32*(float(np.sum(m*da_soft**2))+plate_mass*(acceleration[0,-1]-acceleration[1,-1])**2)
    spring_loss = h*h/32*stiffness*(velocities[0,-1]-velocities[1,-1])**2
    elastic_defect = U1-U0-elastic_work
    physical_balance = dk+U1-U0+spring+material_work+drive_loss-target_work-contact_work
    dP = np.sum(m*(V-v), axis=0)+np.array([0.,0.,plate_mass*(W-w)])
    dL = np.cross(X,m*V).sum(axis=0)-np.cross(x,m*v).sum(axis=0)
    angular_rk = -h*h/16*np.cross(stages[0][1]-stages[1][1],m*da_soft).sum(axis=0)
    minimum_reaction = float(min(lower.min(initial=np.inf),upper.min(initial=np.inf))) if len(bottom)+len(top) else 0.
    info = dict(method='nonlinear-fully-coupled-Radau-IIA-2-stage-order-3', native_acceptance=False,
        material_coordinates='unrounded rest-relative increment; extended-precision polar strain',
        newton_iterations=iteration, linear_iterations=linear_counts, force_residual_N=error,
        residual_history_N=history, material_gamma=material_gamma,
        lower_reactions_N=(B @ lower).tolist(), upper_reactions_N=(B @ upper).tolist(),
        stage_lower_reactions_N=lower.tolist(), stage_upper_reactions_N=upper.tolist(),
        min_reaction_N=minimum_reaction, unilateral_admissible=minimum_reaction>=-1e-8,
        material_energy_change_J=U1-U0, material_force_work_J=elastic_work,
        material_quadrature_defect_J=elastic_defect, kinetic_energy_change_J=dk,
        spring_energy_change_J=spring, target_work_J=target_work,
        material_and_linear_damping_work_J=material_work, drive_dissipation_J=drive_loss,
        contact_work_J=contact_work, kinetic_integration_loss_J=kinetic_loss,
        spring_integration_loss_J=spring_loss, physical_work_defect_J=physical_balance,
        work_accounting_residual_J=physical_balance+kinetic_loss+spring_loss-elastic_defect,
        external_momentum_residual_Ns=float(np.linalg.norm(dP-external_impulse)),
        soft_angular_defect_Nms=(dL-soft_torque).tolist(),
        soft_angular_RK_defect_Nms=angular_rk.tolist(),
        soft_angular_accounting_residual_Nms=float(np.linalg.norm(dL-soft_torque-angular_rk)),
        position_constraint_m=max(maxabs(X[bottom,2]-table),maxabs(X[top,2]-Q)),
        velocity_constraint_m_s=max(maxabs(V[bottom,2]),maxabs(V[top,2]-W)),
        min_J=min(float(model.kinematics(s[0])[1].min()) for s in stages))
    if record_motion:
        relative_initial = w-expand(u0)[:,2]
        relative_acceleration = np.array([acceleration[i,-1]-stages[i][2][:,2] for i in range(2)])
        upper_increment = h*relative_initial+h*h*((B @ A) @ relative_acceleration)
        info.update(integrated_soft_displacement_m=expand(displacements[1]).tolist(),
                    integrated_plate_displacement_m=float(displacements[1,-1]),
                    integrated_upper_gap_increment_m=upper_increment.tolist())
    return X,V,float(Q),float(W),info
