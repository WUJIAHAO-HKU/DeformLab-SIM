"""Additional numerical checks for the shared-cap, physical-accounting and refresh data."""
from pathlib import Path
import collections,json
import numpy as np
DATA=Path(__file__).resolve().parents[1]/'data'
def read(n):return json.loads((DATA/n).read_text(encoding='utf8'))
def shared_cap():
 rows=read('shared-cap/48-runs.json');assert len(rows)==48
 cells=collections.defaultdict(list)
 for r in rows:
  # Prefix/engine/iteration tokens differ; condition, seed and actual gap define a pair.
  gap=r['name'].split('-gap')[-1];cells[(r['condition'],r['seed'],gap)].append(r)
 assert len(cells)==12 and all(len(v)==4 for v in cells.values())
 eligible={k for k,v in cells.items() if all(r['timing']['timing_valid'] for r in v)};assert len(eligible)==11
 out={}
 for method in ['D','cold','dense','VBD']:
  out[method]={}
  for condition in ['narrow_soft_far','wide_stiff_near']:
   group=[r for r in rows if r['backend']==method and r['condition']==condition]
   timed=[r for k,v in cells.items() if k in eligible for r in v if r['backend']==method and r['condition']==condition]
   assert len(group)==6
   out[method][condition]={'pass':sum(r['task_success'] for r in group),'n':6,'hold_mm':float(np.mean([r['hold_mae_m'] for r in group])*1000),'free_mm':float(np.mean([r['withdrawal_mae_m'] for r in group])*1000),'mean_loop_s':float(np.mean([r['loop_wall_s'] for r in timed]))}
 assert [sum(v['pass'] for v in out[m].values()) for m in out]==[10,10,10,11]
 return {'methods':out,'common_timing_cells':len(eligible)}
def physical_accounting():
 energy=read('physics/energy-refinement.json');coupon_input=read('physics/coupon-reference.json');force=read('physics/known-load-and-momentum.json')
 phases=energy['phases_by_rate'];assert [r['hz'] for r in phases]==[480,960,1920]
 out=[]
 for rate in phases:
  whole=next(r for r in rate['phases'] if r['phase']=='whole');q=whole['joint'];balance=q['delta_Ksoft_J']+q['delta_Kart_J']+q['delta_U_J']-q['actuator_boundary_work_J']
  assert np.isclose(balance,q['physical_target_residual']['signed_sum_J'],atol=1e-13)
  rev=next(r for r in rate['phases'] if r['phase']=='unloading')
  out.append({'hz':rate['hz'],'signed_energy_residual_uJ':balance*1e6,'copy_loss_uJ':whole['copy_variance_loss_J']['sum_absolute']*1e6,'reverse_local_defect_uJ':rev['material_local_target_plus_native_work_partition_sumabs_J']*1e6})
 assert np.allclose([r['signed_energy_residual_uJ'] for r in out],[-1.760,-.806,-.317],atol=.00051)
 assert np.allclose([r['copy_loss_uJ'] for r in out],[22.465,9.501,4.755],atol=.00051)
 maximum=max(r['momentum_accounting']['maximum_frame_residual_norm_Ns'] for r in force);assert maximum<8.4e-11
 deformation=np.array(energy['common_grid_curve_differences']['soft_deformation_m']['adjacent_rms_differences'])*1e6;assert np.allclose(deformation,[8.80,5.00],atol=.005)
 coupon=coupon_input['aggregate_numbers'];assert coupon['D960_force_shape_percent']==5.7 and coupon['quasistatic_joint_gate_passes']==0
 return {'rates':out,'maximum_momentum_residual_Ns':maximum,'successive_rate_shape_difference_um':deformation.tolist(),'coupon':coupon}
def refresh_policy():
 a=read('refresh/plot-data.json');independent=read('refresh/independent-runs.json');out=[]
 for ci in [0,1]:
  with np.load(DATA/'refresh/material-reference'/f'c{ci}.npz',allow_pickle=False) as z:
   assert z['rest'].shape==(2197,3) and z['elements'].shape==(10368,4)
   assert np.isfinite(z['gain_xy']).all() and (z['gain_xy']>=0).all()
   out.append({'material_pa':float(z['young_pa']),'response_gain_min':float(z['gain_xy'].min()),'response_gain_max':float(z['gain_xy'].max())})
 pairs=np.array(a['independent_D_prefinal_margins_um']);assert pairs.shape==(8,2) and (pairs>0).all()
 return {'material_references':out,'paired_margins_um':pairs.tolist(),'recorded_independent_entries':len(independent)}

def reported_shape():
    data=read('physics/loaded-shape-table.json')
    assert data['methods']['DeformLabSim']['shape_rms']==1.329
    assert data['methods']['Consistent State Mapping']['shape_rms']==13.803
    return data
