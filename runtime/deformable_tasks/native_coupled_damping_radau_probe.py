"""One actual paired velocity publication using stage-constrained Radau IIA damping.

Full measured mass/Jacobian and explicit stage work plus nonnegative RK loss.
Dense CPU diagnostic; not continuous-frame/material/actuator acceptance.
"""
import json
import numpy as np
from .native_coupled_damping_probe import NativeCoupledDampingProbe as Prototype
from .damping_contact_coordinates import ContactCoordinates

class NativeCoupledDampingProbe(Prototype):
    def __init__(self,root,base,port,runtime,destination,step=128,substeps=64):
        super().__init__(root,base,port,runtime,destination,step,substeps)
        from contact_damping_radau_reference import integrate
        self.refine=integrate

    def apply(self):
        step=self.runtime.steps+1
        if step!=self.step:
            self.base.apply();self.base.ledger[-1]['step']=step;return
        from isaacsim.core.experimental.utils.backend import use_backend
        import warp as wp
        base=self.base;torch=base.torch;port=self.port.current
        if port['step']!=step or not port['public_link_matches_native']:raise RuntimeError('Fresh matched native link contact data required')
        x=base.p[:,:3].double().cpu().numpy();v=base.v[:,:3].double().cpu().numpy();mass=base.mass.cpu().numpy()
        rows=np.asarray(port['rows'],float).reshape(-1,6);drive=self.runtime.drives[0];tensor=drive._physics_articulation_view
        tool=drive.link_names.index('Tool')
        with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):
            M=drive.get_mass_matrices().numpy()[0].astype(float);qv=drive.get_dof_velocities().numpy()[0].astype(float)
        link=tensor.get_link_velocities().numpy()[0,tool].astype(float);J=np.asarray(port['tool_jacobian'])
        if drive.dof_names!=['DriveX','DriveY','DriveZ'] or J.shape!=(6,3) or np.any(J[3:]!=0) or np.any(link[3:]!=0):raise RuntimeError('Pinned translating Cartesian topology required')
        if not len(rows) or not np.array_equal(qv,port['rigid_velocity_m_s']) or not np.array_equal(link,port['public_link_velocity_m_s']):raise RuntimeError('Native endpoint changed before damping')
        coordinates=ContactCoordinates(M,J[:3],qv,link[:3])
        m=np.r_[mass.repeat(3),np.ones(3)];v0=np.r_[v.ravel(),coordinates.velocity]
        D,error=self.operator(self.material,x,base.gamma);aug=np.zeros((len(m),len(m)));aug[:-3,:-3]=D
        groups={}
        for r in rows:groups.setdefault((int(r[0]),int(r[1])),[]).append(r)
        C=np.zeros((len(groups),len(m)));bounds=[];offsets=[];contacts=[]
        for i,((node,top),rs) in enumerate(sorted(groups.items())):
            C[i,3*node+2]=-1. if top else 1.;offset=0.
            if top:C[i,-3:],offset=coordinates.row([0,0,1])
            gap=min(r[2] for r in rs);bound=-max(gap,0.)/port['inner_dt_s']
            bounds.append(bound-offset);offsets.append(offset)
            if max(abs(C[i]@v0+offset-r[3]) for r in rs)>1e-12:raise RuntimeError('Mapped contact endpoint mismatch')
            contacts.append(dict(node=node,top=bool(top),gap_m=gap,source_rows=len(rs),physical_bound_m_s=bound,affine_readout_residual_m_s=offset))
        bounds=np.asarray(bounds);offsets=np.asarray(offsets);dt=1/self.runtime.config['solver']['hz']
        solution=self.refine(m,aug,v0,C,bounds,dt,self.substeps);new=solution['velocity'];qnew=coordinates.joint_state(new[-3:])
        softpub=new[:-3].reshape(-1,3).astype(np.float32).astype(float);qpub=qnew.astype(np.float32).astype(float)
        upub=np.linalg.solve(coordinates.to_joint,qpub);published=np.r_[softpub.ravel(),upub]
        impulses=solution['total_contact_impulse'];top=sum(impulses[i] for i,c in enumerate(contacts) if c['top']);bottom=sum(impulses[i] for i,c in enumerate(contacts) if not c['top'])
        world_impulse=np.array([0.,0.,top]);joint_impulse=coordinates.generalized_impulse(world_impulse)
        joint_error=M@(qnew-qv)-joint_impulse
        soft_error=(mass[:,None]*(new[:-3].reshape(-1,3)-v)).sum(0)-np.array([0.,0.,bottom-top])
        if solution['stage_equation_residual']>1e-9 or solution['peak_momentum_equation_residual']>1e-10 or max(np.linalg.norm(joint_error),np.linalg.norm(soft_error))>1e-9 or solution['peak_velocity_violation']>1e-9 or abs(solution['energy_balance_residual'])>1e-10:raise RuntimeError('Coupled mapped impulse/work solve failed')
        path=self.dest/'native-coupled-damping-publication-v4.json'
        if path.exists():raise ValueError('Fresh one-shot evidence required')
        self.record=dict(step=step,status='prepared',scope=__doc__,substeps=self.substeps,material_gamma=base.gamma,
            method=solution['method'],stage_equation_residual_m_s=solution['stage_equation_residual'],
            momentum_equation_residual_Ns=solution['peak_momentum_equation_residual'],max_lcp_iterations=max(solution['lcp_iterations']),
            measured_generalized_mass_kg=M.tolist(),tool_jacobian=J.tolist(),affine_link_residual_m_s=coordinates.observation_residual.tolist(),
            contacts=contacts,contact_multipliers_Ns=impulses.tolist(),independent_operator_relative_error=error,
            joint_velocity_before_m_s=qv.tolist(),joint_velocity_intended_m_s=qpub.tolist(),tool_velocity_before_m_s=link.tolist(),
            predicted_tool_velocity_m_s=(J[:3]@qpub+coordinates.observation_residual).tolist(),
            world_contact_impulse_on_tool_Ns=world_impulse.tolist(),generalized_contact_impulse_Ns=joint_impulse.tolist(),
            generalized_impulse_error_Ns=joint_error.tolist(),soft_resultant_impulse_error_Ns=soft_error.tolist(),
            kinetic_change_J=solution['kinetic_change'],stage_damping_work_J=solution['stage_damping_work'],
            integration_dissipation_J=solution['integration_dissipation'],generalized_contact_stage_work_J=solution['contact_stage_work'],
            affine_readout_work_J=float(impulses@offsets),world_contact_stage_work_J=float(solution['contact_stage_work']+impulses@offsets),
            generalized_work_balance_residual_J=solution['energy_balance_residual'],
            publication_energy_change_J=float(.5*np.dot(m,published*published-new*new)),
            reference_velocity_violation_m_s=solution['peak_velocity_violation'],
            free_Radau_velocity_violation_m_s=float(np.maximum(bounds-C@solution['free_velocity'],0).max(initial=0)),
            predicted_published_velocity_violation_m_s=float(np.maximum(bounds-C@published,0).max(initial=0)),
            native_intervention=True,platform_acceptance=False,actuator_work_closed=False,whole_system_angular_balance_accepted=False,
            readback_threshold_m_s=1e-9)
        def save():path.write_text(json.dumps(self.record,indent=2,allow_nan=False)+'\n')
        save()
        np.savez_compressed(self.dest/'native-coupled-damping-states-v4.npz',positions=x,soft_mass=mass,generalized_mass=M,jacobian=J,
            soft_before=v,soft_intended=new[:-3].reshape(-1,3),soft_published=softpub,joint_before=qv,joint_intended=qnew,joint_published=qpub,
            transformed_mass=m,transformed_before=v0,transformed_intended=new,transformed_published=published,constraint=C,bound=bounds,affine_offset=offsets)
        target_v=tensor.get_dof_velocity_targets().numpy().copy();target_q=tensor.get_dof_position_targets().numpy().copy()
        qposition=tensor.get_dof_positions().numpy().copy()
        indices=wp.array([0],dtype=wp.int32,device='cuda:0');data=wp.array(qpub.astype(np.float32).reshape(1,3),dtype=wp.float32,device='cuda:0')
        from isaacsim.core.simulation_manager import SimulationManager
        link_pose_before=tensor.get_link_transforms().numpy().copy();clock_before=SimulationManager.get_simulation_time()
        import ctypes as ct
        old_links=tensor.get_link_velocities().numpy()[0].copy()
        all_jac=tensor.get_jacobians().numpy()[0].astype(float)
        if all_jac.shape!=(3,6,3) or np.any(all_jac[:,3:]!=0) or np.any(old_links[:,3:]!=0):raise RuntimeError('Unsupported native velocity map')
        new_links=np.zeros((4,3),np.float32)
        for i in range(1,4):new_links[i]=(all_jac[i-1,:3]@qpub).astype(np.float32)
        history_before=tensor.get_link_incoming_joint_force().numpy().copy()
        acceleration_before=tensor.get_link_accelerations().numpy().copy()
        with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):tau_before=drive.get_dof_projected_joint_forces().numpy().copy()
        P=ct.POINTER(ct.c_float);o=self.port.o
        o.observer_damping_velocity_publish.argtypes=[ct.c_ulonglong,*([P]*6),ct.c_uint];o.observer_damping_velocity_publish.restype=ct.c_int
        o.observer_damping_velocity_count.restype=ct.c_ulonglong
        arrays=[np.ascontiguousarray(a,dtype=np.float32) for a in (qv,qpub,old_links[:,:3],new_links,v,softpub)]
        code=o.observer_damping_velocity_publish(step,*[a.ctypes.data_as(P) for a in arrays],len(mass))
        self.record['native_velocity_write_code']=int(code);self.record['native_velocity_publications']=int(o.observer_damping_velocity_count());save()
        if code:raise RuntimeError('Native velocity-only transaction rejected '+str(code))
        torch.cuda.synchronize()
        actual=base.v[:,:3].double().cpu().numpy();actualq=tensor.get_dof_velocities().numpy()[0].astype(float)
        actual_link=tensor.get_link_velocities().numpy()[0,tool].astype(float)
        actual_relative=np.array([(actual_link[2]-actual[c['node'],2]) if c['top'] else actual[c['node'],2] for c in contacts])
        actual_bound=np.array([c['physical_bound_m_s'] for c in contacts])
        history_after=tensor.get_link_incoming_joint_force().numpy().copy();acceleration_after=tensor.get_link_accelerations().numpy().copy()
        with use_backend('tensor',raise_on_fallback=True,raise_on_unsupported=True):tau_after=drive.get_dof_projected_joint_forces().numpy().copy()
        self.record.update(status='published',
            native_force_history_unchanged=bool(np.array_equal(history_before,history_after)),
            native_acceleration_history_unchanged=bool(np.array_equal(acceleration_before,acceleration_after)),
            native_drive_effort_unchanged=bool(np.array_equal(tau_before,tau_after)),actual_joint_velocity_m_s=actualq.tolist(),actual_tool_velocity_m_s=actual_link.tolist(),
            velocity_targets_unchanged=bool(np.array_equal(target_v,tensor.get_dof_velocity_targets().numpy())),
            position_targets_unchanged=bool(np.array_equal(target_q,tensor.get_dof_position_targets().numpy())),
            joint_positions_unchanged=bool(np.array_equal(qposition,tensor.get_dof_positions().numpy())),
            soft_positions_unchanged=bool(np.array_equal(x,base.p[:,:3].double().cpu().numpy())),
            soft_publication_error_m_s=float(abs(actual-softpub).max()),joint_publication_error_m_s=float(abs(actualq-qpub).max()),
            link_publication_error_m_s=float(abs(actual_link[:3]-J[:3]@qpub).max()),
            affine_model_prediction_error_m_s=float(abs(actual_link[:3]-(J[:3]@qpub+coordinates.observation_residual)).max()),
            link_positions_unchanged=bool(np.array_equal(link_pose_before,tensor.get_link_transforms().numpy())),
            maximum_link_pose_refresh_change=float(abs(link_pose_before-tensor.get_link_transforms().numpy()).max()),
            simulation_clock_unchanged=bool(clock_before==SimulationManager.get_simulation_time()),
            actual_contact_velocity_violation_m_s=float(np.maximum(actual_bound-actual_relative,0).max(initial=0)))
        self.record['passed']=all(self.record[k] for k in ('velocity_targets_unchanged','position_targets_unchanged','joint_positions_unchanged','soft_positions_unchanged','link_positions_unchanged','simulation_clock_unchanged','native_force_history_unchanged','native_acceleration_history_unchanged','native_drive_effort_unchanged')) and max(self.record[k] for k in ('soft_publication_error_m_s','joint_publication_error_m_s','link_publication_error_m_s','actual_contact_velocity_violation_m_s'))<=1e-9
        np.savez_compressed(self.dest/'native-coupled-damping-observed-v4.npz',soft_actual=actual,joint_actual=actualq,link_actual=actual_link,
            velocity_target_before=target_v,velocity_target_after=tensor.get_dof_velocity_targets().numpy(),position_target_before=target_q,position_target_after=tensor.get_dof_position_targets().numpy(),
            joint_position_before=qposition,joint_position_after=tensor.get_dof_positions().numpy(),link_pose_before=link_pose_before,link_pose_after=tensor.get_link_transforms().numpy(),
            native_force_history_before=history_before,native_force_history_after=history_after,native_acceleration_history_before=acceleration_before,native_acceleration_history_after=acceleration_after,
            native_drive_effort_before=tau_before,native_drive_effort_after=tau_after,all_link_jacobians=all_jac,all_link_velocity_before=old_links,all_link_velocity_intended=new_links,all_link_velocity_after=tensor.get_link_velocities().numpy()[0])
        self.record['next_native_cache_verified']=False;save();self.publication_path=path;self.expected_q=qpub.copy();self.expected_soft=softpub.copy();self.expected_target_v=target_v
        if not self.record['passed']:raise RuntimeError('Mapped native paired publication failed; actual evidence retained')

    def before_native(self,step):
        if step!=self.step+1 or self.record is None:return
        tensor=self.runtime.drives[0]._physics_articulation_view
        soft=self.base.v[:,:3].double().cpu().numpy();joint=tensor.get_dof_velocities().numpy()[0].astype(float)
        self.record['next_frame_before_native']=dict(step=step,soft_velocity_unchanged=bool(np.array_equal(soft,self.expected_soft)),joint_velocity_unchanged=bool(np.array_equal(joint,self.expected_q)),velocity_targets_unchanged=bool(np.array_equal(self.expected_target_v,tensor.get_dof_velocity_targets().numpy())))
        self.publication_path.write_text(json.dumps(self.record,indent=2)+'\n')
        if not all(v for k,v in self.record['next_frame_before_native'].items() if k!='step'):raise RuntimeError('Paired velocity lost before next native step')

    def report(self):
        other=self.base.report();other['accepted']=self.base.steps==self.runtime.steps-1
        return dict(accepted=bool(self.record and self.record['passed'] and other['accepted']),steps=self.runtime.steps,
            integrator='one-selected-full-mass-Jacobian-coupled-Radau-step',intervention_steps=[self.step],reference_substeps=self.substeps,
            requested_gamma=self.base.gamma,exponential_other_frames=other,publication=self.record,scope=__doc__,platform_acceptance=False)
