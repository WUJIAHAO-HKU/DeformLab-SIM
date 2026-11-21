"""Run a bounded linear FEM whole-step reference through the platform CLI.

CPU correctness reference; timings are not comparable to native GPU backends.
The native rest mesh is shared, but assembly/integration/contact are independent.
"""
import argparse,csv,hashlib,json,sys,time,traceback,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from deformable_tasks.coupled_coupon_reference import CoupledCoupon,ImpactCoupledCoupon
from deformable_tasks.adaptive_coupon_reference import AdaptiveCoupledCoupon
from deformable_tasks.midpoint_coupon_reference import HybridMidpointCoupon
from deformable_tasks.progress_logging import emit_json
from deformable_tasks.hybrid_gauss_coupon import HybridGaussCoupon

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def smooth(x):
    x=np.clip(x,0.,1.);return x*x*x*(10+x*(-15+6*x))
def target_at(t,L,compression):return -L*compression*smooth((t-.4)/1.4)*(1-smooth((t-2.7)/.7))
def target_rate_at(t,L,compression):
    x=(t-.4)/1.4;y=(t-2.7)/.7
    derivative=lambda z:30*z*z*(1-z)**2 if 0<z<1 else 0.
    return -L*compression*(derivative(x)/1.4*(1-smooth(y))-smooth(x)*derivative(y)/.7)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-case',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--hz',type=int,default=960);p.add_argument('--duration',type=float,default=4.)
    p.add_argument('--kelvin-voigt-s',type=float,default=0.)
    p.add_argument('--impact-projection',action='store_true',help='Add separately audited endpoint inelastic velocity impulse')
    p.add_argument('--adaptive',action='store_true',help='Error-controlled dyadic steps with separately audited impact')
    p.add_argument('--velocity-tolerance',type=float,default=2e-5)
    p.add_argument('--position-tolerance',type=float,default=1e-8)
    p.add_argument('--integrator',choices=('backward-euler','midpoint-hybrid','gauss-hybrid'),default='backward-euler')
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh output directory; existing evidence is preserved')
    source=Path(a.source_case);config=json.loads((source/'effective-task.json').read_text())
    if config['task']!='soft_squeeze' or config['material']['friction']!=0:raise ValueError('Only frictionless soft_squeeze is supported')
    if not (a.hz>0 and a.hz%15==0 and a.duration>=3.5 and a.kelvin_voigt_s>=0):raise ValueError('Require hz divisible by 15 and full release horizon >=3.5s')
    data=np.load(source/'trajectory.npz');rest=data['rest_0'].astype(float);tets=data['elements_0'].astype(int)
    mat=config['material'];L=config['geometry']['size_m'];table=.12
    if not np.allclose(np.ptp(rest,axis=0),L,atol=1e-7,rtol=0):raise ValueError('Source mesh is not the expected cube')
    if not np.isclose(rest[:,2].min(),table+.0005,atol=1e-7,rtol=0):raise ValueError('Source rest mesh is not in expected world coordinates')
    method=('Hybrid implicit midpoint with backward-Euler activation steps' if a.integrator=='midpoint-hybrid' else 'Whole-step backward Euler')
    if a.integrator=='gauss-hybrid':method='Gauss smooth phases with consistent Hermite command; midpoint/backward-Euler contact transitions'
    meta=dict(method=method+', linear tetrahedral FEM, dynamic vertical platen and unilateral frictionless contact',
        scope='Independent CPU small-strain diagnostic; shared native rest mesh; NOT native finite-strain validation, all-material acceptance, or same-budget comparison',
        impact_projection=a.impact_projection or a.adaptive or a.integrator in ('midpoint-hybrid','gauss-hybrid'),adaptive=a.adaptive,integrator=a.integrator,
        velocity_tolerance_m_s=a.velocity_tolerance,position_tolerance_m=a.position_tolerance,
        source_case=str(source),source_sha256={f:sha(source/f) for f in ('effective-task.json','trajectory.npz')},
        source_config=config,hz=a.hz,duration_s=a.duration,kelvin_voigt_s=a.kelvin_voigt_s,
        damping_scope='D_soft = linear_damping*M + kelvin_voigt_s*K. Kelvin-Voigt coefficient is not native elasticity gamma.',
        articulation_scope='Vertical DOF of fixed-base three-prismatic Cartesian articulation; X/Y decouple for frictionless normal contact; NOT general articulated robot',
        platen_mass_kg=.2,drive_stiffness_N_m=1e6,drive_damping_N_s_m=4000.,table_z_m=table,initial_plate_bottom_m=table+L+.002,
        source_sha256_code={str(f.relative_to(ROOT)):sha(f) for f in (Path(__file__),*[ROOT/'deformable_tasks'/name for name in ('coupled_coupon_reference.py','adaptive_coupon_reference.py','midpoint_coupon_reference.py','progress_logging.py','hybrid_gauss_coupon.py','driven_gauss_reference.py','frozen_gauss_reference.py','frozen_contact_oracle.py','gauss_factor.py')])})
    (out/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    for rel in meta['source_sha256_code']:
        dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    cls=ImpactCoupledCoupon if a.impact_projection else CoupledCoupon
    if a.integrator=='midpoint-hybrid':cls=HybridMidpointCoupon
    if a.integrator=='gauss-hybrid':cls=HybridGaussCoupon
    extra={}
    if a.adaptive:
        cls=AdaptiveCoupledCoupon
        extra=dict(target_function=lambda t:target_at(t,L,config['action']['compression']),position_tolerance=a.position_tolerance,velocity_tolerance=a.velocity_tolerance,integrator=a.integrator)
        if a.integrator=='gauss-hybrid':extra['target_derivative']=lambda t:target_rate_at(t,L,config['action']['compression'])
    sim=cls(rest,tets,mat['young_pa'],mat['poisson'],mat['density'],a.hz,
        linear_damping=config['solver']['linear_damping'],kelvin_voigt_s=a.kelvin_voigt_s,table_z=table,plate_bottom=table+L+.002,**extra)
    rows=[];states=[rest.copy()];velocities=[np.zeros_like(rest)];times=[0.];tool=[0.];start=time.perf_counter();failure=None
    subfile=(out/'substeps.csv').open('w',newline='') if a.adaptive else None;subwriter=None
    try:
        for step in range(round(a.duration*a.hz)):
            t=(step+1)/a.hz
            if a.integrator=='gauss-hybrid' and not a.adaptive:
                rows.append(sim.step(target_at(t,L,config['action']['compression']),target_rate_at(t-1/a.hz,L,config['action']['compression']),target_rate_at(t,L,config['action']['compression'])))
            else:rows.append(sim.step(target_at(t,L,config['action']['compression'])))
            if a.adaptive:
                if subwriter is None:
                    subwriter=csv.DictWriter(subfile,fieldnames=list(sim.last_substeps[0]));subwriter.writeheader()
                subwriter.writerows(sim.last_substeps)
            if (step+1)%(a.hz//15)==0:
                states.append(rest+sim.u[:-1].reshape(-1,3));velocities.append(sim.v[:-1].reshape(-1,3).copy());times.append(t);tool.append(sim.u[-1])
            if (step+1)%a.hz==0:emit_json(dict(time_s=t,elapsed_s=time.perf_counter()-start,**{k:rows[-1][k] for k in ('reaction_N','max_nodal_speed_m_s','max_penetration_m')}),out/'progress.jsonl')
    except Exception:
        failure=traceback.format_exc();(out/'failure.txt').write_text(failure)
        if hasattr(sim,'failure_state'):np.savez_compressed(out/'failed-trial.npz',**sim.failure_state)
    finally:
        if subfile:subfile.close()
    if rows:
        with (out/'steps.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(out/'trajectory.npz',time_s=times,nodes_0=states,velocities_0=velocities,tool_q_m=tool,rest_0=rest,elements_0=tets,mass_0=sim.mass)
    hold=[r for r in rows if 2.1<=r['time_s']<=2.5];E=mat['young_pa'];area=L*L
    gates=dict(completed=failure is None and len(rows)==round(a.duration*a.hz),hold_sampled=bool(hold))
    summary=dict(scope=meta['scope'],hz=a.hz,impact_projection=meta['impact_projection'],adaptive=a.adaptive,integrator=a.integrator,kelvin_voigt_s=a.kelvin_voigt_s,steps=len(rows),elapsed_s=time.perf_counter()-start,
        nodes=len(rest),elements=len(tets),mass_kg=float(sim.mass.sum()),expected_mass_kg=mat['density']*L**3,
        metrics={},gates=gates)
    if rows:
        m=summary['metrics']
        for k in ('max_penetration_m','max_nodal_speed_m_s','force_residual_N','natural_residual_m','iterations'):
            m[k]=max(r[k] for r in rows)
        m['min_J']=min(r['min_J'] for r in rows)
        for k in ('mechanical_work_residual_J','full_work_residual_J'):
            m['max_abs_'+k]=max(abs(r[k]) for r in rows);m['sum_'+k]=sum(r[k] for r in rows)
        for k in ('drive_work_J','contact_work_J','material_drag_dissipation_J','integration_loss_J','drive_target_work_J','drive_dissipation_J','total_integration_loss_J'):
            m['sum_'+k]=sum(r[k] for r in rows)
        if hold:
            m['hold_peak_speed_m_s']=max(r.get('peak_substep_nodal_speed_m_s',r['max_nodal_speed_m_s']) for r in hold)
            m['hold_mean_reaction_N']=float(np.mean([r['reaction_N'] for r in hold]));m['hold_mean_strain']=float(np.mean([r['strain'] for r in hold]))
            m['hold_force_relative_error']=abs(m['hold_mean_reaction_N']-E*area*m['hold_mean_strain'])/max(abs(E*area*m['hold_mean_strain']),1e-12)
            gates.update(quasistatic=m['hold_peak_speed_m_s']<=.002,linear_force_accuracy=m['hold_force_relative_error']<=.05)
        gates.update(contact=m['max_penetration_m']<=1e-9,positive_volume=m['min_J']>0,
            force_equilibrium=m['force_residual_N']<=1e-6,work_accounting=m['max_abs_full_work_residual_J']<=1e-9)
        if meta['impact_projection']:
            for k in ('impact_impulse_residual_Ns','closed_inward_speed_m_s','position_integrated_velocity_residual_m','linearized_angular_impulse_residual_Nms'):
                m['max_'+k]=max(abs(r[k]) for r in rows)
            m['sum_impact_work_J']=sum(r['impact_work_J'] for r in rows)
            gates.update(impact_velocity=m['max_closed_inward_speed_m_s']<=1e-9,
                angular_impulse=m['max_linearized_angular_impulse_residual_Nms']<=1e-10)
        if a.adaptive:
            summary.update(accepted_substeps=sim.total_accepted,rejected_trials=sim.total_rejected)
            m.update(minimum_substep_s=min(r['minimum_substep_s'] for r in rows),max_local_error=max(r['max_accepted_error'] for r in rows))
        if a.integrator in ('midpoint-hybrid','gauss-hybrid'):
            m['midpoint_steps']=sum(r['midpoint_step'] for r in rows)
            m['contact_activation_fallbacks']=sum(r['contact_activation_fallback'] for r in rows)
        if a.integrator=='gauss-hybrid':
            for k in ('gauss_step','contact_phase_fallback'):m[k]=sum(r[k] for r in rows)
            for k in ('gauss_mapping_position_error_m','gauss_mapping_velocity_error_m_s','gauss_impulse_residual_Ns'):m['max_'+k]=max(r[k] for r in rows)
    gates.update({key:bool(value) for key,value in gates.items()})
    summary['passed']=all(gates.values());(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');emit_json(summary,out/'progress.jsonl')
    if failure:print(failure,file=sys.stderr)
    return 0 if summary['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
