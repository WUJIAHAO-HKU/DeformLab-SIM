"""Native completion is a receipt contract, never inferred from raw OS exit0."""
def completion(native,raw_exit_code,config):
    expected=round(config['duration_s']*config['solver']['hz'])
    checks=dict(raw_exit_zero=raw_exit_code==0,completed_status=native.get('status')=='completed_pending_exit',semantic_exit_zero=native.get('exit_code')==0,no_cleanup_errors=not native.get('errors'),all_steps=native.get('completed_steps')==expected)
    return dict(passed=all(checks.values()),checks=checks,status=native.get('status'),semantic_exit_code=native.get('exit_code'),error=native.get('error'),traceback=native.get('traceback'),requested_steps=expected,completed_steps=native.get('completed_steps'))
