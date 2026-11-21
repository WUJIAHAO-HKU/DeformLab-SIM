"""Conservative comparison eligibility: work count is not elapsed-time budget."""
import copy

def physical_signature(config):
    c=copy.deepcopy(config)
    c['solver'].pop('hz');c['solver'].pop('position_iterations')
    return c

def compare(a,b,tolerance=.10,clean_gpu=True):
    same=physical_signature(a['config'])==physical_signature(b['config'])
    complete=a['completed_duration'] and b['completed_duration']
    ratio=b['timing']['compute_s_per_sim_second']/a['timing']['compute_s_per_sim_second']
    valid=all(a['checks'].values()) and all(b['checks'].values())
    matched=abs(ratio-1)<=tolerance
    return dict(physical_configuration_equal=same,full_horizons_completed=complete,
        all_physical_checks_passed=valid,compute_time_ratio_candidate_over_control=ratio,
        time_budget_within_tolerance=matched,gpu_observation_clean=clean_gpu,
        budget_pair_eligible=bool(same and complete and matched and clean_gpu),
        physical_accuracy_claim_eligible=bool(same and complete and matched and clean_gpu and valid),
        note='A failed complete run is still a valid failure-rate observation. No accuracy or speedup claim from an early-aborted rollout.')

def select_iterations(target,low,high,low_iterations=1,high_iterations=32):
    if high<=low:return low_iterations,'nonmonotonic calibration; lower endpoint only'
    n=round(low_iterations+(target-low)*(high_iterations-low_iterations)/(high-low))
    return max(1,min(255,n)),'linear interpolation/extrapolation of compute time only; full-rollout timing must validate'
