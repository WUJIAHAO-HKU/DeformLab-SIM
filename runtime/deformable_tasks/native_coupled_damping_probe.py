"""One selected native coupled-damping publication, followed by ordinary evolution.

Diagnostic backward-Euler substeps; explicit integration loss, not an
exponential material-equivalence or whole-horizon solver acceptance.
"""
import json,sys
import numpy as np

class NativeCoupledDampingProbe:
    def __init__(self,root,base,port,runtime,destination,step=128,substeps=4):
        self.base=base;self.port=port;self.runtime=runtime;self.dest=destination
        self.step=step;self.substeps=substeps;self.record=None
        sys.path.insert(0,str(root/'scripts'))
        from probe_coupled_damping_reference import operator
        from refine_contact_damping_reference import refine
        from deformable_tasks.corotated_material_reference import CorotatedMaterial
        self.operator=operator;self.refine=refine
        b=runtime.bodies[0];cfg=runtime.config
        self.material=CorotatedMaterial(b['rest'],b['sim_elements'],cfg['material']['young_pa'],cfg['material']['poisson'])

    def apply(self):
        step=self.runtime.steps+1
        if step!=self.step:
            self.base.apply()
            self.base.ledger[-1]['step']=step
            return
        from isaacsim.core.experimental.utils.backend import use_backend
        import warp as wp
        base=self.base;torch=base.torch;port=self.port.current
        if port['step']!=step:raise RuntimeError('Fresh terminal contact token required')
        x=base.p[:,:3].double().cpu().numpy();v=base.v[:,:3].double().cpu().numpy()
        mass=base.mass.cpu().numpy();rows=np.asarray(port['rows'],float).reshape(-1,6)
        if not len(rows):raise RuntimeError('Coupled write probe requires actual native contacts')
        drive=self.runtime.drives[0]
        with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
            M=drive.get_mass_matrices().numpy()[0].astype(float)
            rv=drive.get_dof_velocities().numpy()[0].astype(float)
        if drive.dof_names!=['DriveX','DriveY','DriveZ'] or not np.allclose(M,np.diag([2.2,1.2,.2]),rtol=1e-6,atol=0):raise RuntimeError('Unsupported joint mass or topology')
        # Use measured generalized masses for both the solve and impulse ledger.
        if not np.array_equal(rv,np.asarray(port['rigid_velocity_m_s'])):raise RuntimeError('Rigid state changed since contact read')
        groups={}
        for r in rows:groups.setdefault((int(r[0]),int(r[1])),[]).append(r)
        m=np.r_[mass.repeat(3),np.diag(M)];v0=np.r_[v.ravel(),rv]
        D,error=self.operator(self.material,x,base.gamma);aug=np.zeros((len(m),len(m)));aug[:-3,:-3]=D
        C=np.zeros((len(groups),len(m)));bound=[];contacts=[]
        for i,((node,top),rs) in enumerate(sorted(groups.items())):
            C[i,3*node+2]=-1. if top else 1.
            if top:C[i,-1]=1.
            gap=min(r[2] for r in rs);bound.append(-max(gap,0.)/port['inner_dt_s'])
            if max(abs(C[i]@v0-r[3]) for r in rs)>1e-12:raise RuntimeError('Contact endpoint mismatch')
            contacts.append(dict(node=node,top=bool(top),gap_m=gap,source_rows=len(rs)))
        dt=1/self.runtime.config['solver']['hz'];solution=self.refine(m,aug,v0,C,bound,dt,self.substeps)
        new=solution['velocity'];published=new.astype(np.float32).astype(float)
        impulses=solution['total_contact_impulse'];static=np.array([0.,0.,sum(impulses[i] for i,c in enumerate(contacts) if not c['top'])])
        dp=(mass[:,None]*(new[:-3].reshape(-1,3)-v)).sum(0)+np.diag(M)*(new[-3:]-rv)
        if np.linalg.norm(dp-static)>1e-9 or solution['peak_velocity_violation']>1e-9 or abs(solution['energy_balance_residual'])>1e-10:raise RuntimeError('Coupled reference balance failed before publication')
        slack=C@published-np.asarray(bound)
        record=dict(step=step,scope=__doc__,status='prepared',substeps=self.substeps,material_gamma=base.gamma,
            contact_rows=len(rows),contact_groups=len(groups),contacts=contacts,independent_operator_relative_error=error,
            rigid_velocity_before_m_s=rv.tolist(),rigid_velocity_intended_m_s=published[-3:].tolist(),
            additional_rigid_contact_impulse_Ns=(np.diag(M)*(new[-3:]-rv)).tolist(),
            contact_multipliers_Ns=impulses.tolist(),static_contact_impulse_Ns=static.tolist(),
            linear_momentum_minus_static_impulse_Ns=(dp-static).tolist(),
            integration_dissipation_J=solution['integration_dissipation'],endpoint_damping_work_J=solution['endpoint_damping_work'],
            contact_endpoint_work_J=solution['contact_endpoint_work'],reference_energy_balance_residual_J=solution['energy_balance_residual'],
            publication_energy_change_J=float(.5*np.dot(m,published*published-new*new)),
            predicted_published_velocity_violation_m_s=float(np.maximum(-slack,0).max(initial=0)),
            free_BE_velocity_violation_m_s=float(np.maximum(np.asarray(bound)-C@solution['free_velocity'],0).max(initial=0)),
            native_intervention=True,platform_acceptance=False,actuator_work_closed=False,whole_system_angular_balance_accepted=False)
        path=self.dest/'native-coupled-damping-publication.json'
        if path.exists():raise ValueError('One-shot publication evidence must be fresh')
        path.write_text(json.dumps(record,indent=2)+'\n')
        np.savez_compressed(self.dest/'native-coupled-damping-states.npz',positions=x,mass=m,before=v0,intended=new,published=published,constraint=C,bound=bound)
        # The high-level set_dof_velocities also writes velocity targets in this
        # pinned version. Call the tensor setter only; verify both target arrays.
        tensor=drive._physics_articulation_view
        target_v=tensor.get_dof_velocity_targets().numpy().copy();target_q=tensor.get_dof_position_targets().numpy().copy()
        indices=wp.array([0],dtype=wp.int32,device='cuda:0')
        data=wp.array(published[-3:].astype(np.float32).reshape(1,3),dtype=wp.float32,device='cuda:0')
        tensor.set_dof_velocities(data,indices)
        base.v[:,:3].copy_(torch.as_tensor(published[:-3].reshape(-1,3),device='cuda',dtype=torch.float32));torch.cuda.synchronize()
        actual=base.v[:,:3].double().cpu().numpy()
        actual_rigid=tensor.get_dof_velocities().numpy()[0].astype(float)
        actual_links=tensor.get_link_velocities().numpy()[0]
        tool=drive.link_names.index('Tool')
        record.update(status='published',actual_rigid_velocity_m_s=actual_rigid.tolist(),actual_tool_linear_velocity_m_s=actual_links[tool,:3].astype(float).tolist(),
            velocity_targets_unchanged=bool(np.array_equal(target_v,tensor.get_dof_velocity_targets().numpy())),
            position_targets_unchanged=bool(np.array_equal(target_q,tensor.get_dof_position_targets().numpy())),
            soft_publication_error_m_s=float(abs(actual-published[:-3].reshape(-1,3)).max()),
            rigid_publication_error_m_s=float(abs(actual_rigid-published[-3:]).max()),
            link_publication_error_m_s=float(abs(actual_links[tool,:3]-published[-3:]).max()))
        record['passed']=record['velocity_targets_unchanged'] and record['position_targets_unchanged'] and max(record[k] for k in ('soft_publication_error_m_s','rigid_publication_error_m_s','link_publication_error_m_s'))<=1e-9
        path.write_text(json.dumps(record,indent=2)+'\n');self.record=record
        if not record['passed']:raise RuntimeError('Native two-sided publication failed; partial-state evidence retained')

    def report(self):
        base=self.base.report();base['accepted']=self.base.steps==self.runtime.steps-1
        return dict(accepted=bool(self.record and self.record['passed'] and base['accepted']),
            steps=self.runtime.steps,integrator='one-selected-coupled-BE-step-with-exponential-other-frames',
            intervention_steps=[self.step],reference_substeps=self.substeps,requested_gamma=self.base.gamma,
            exponential_other_frames=base,publication=self.record,scope=__doc__,platform_acceptance=False)
