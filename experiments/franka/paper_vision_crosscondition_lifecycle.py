"""Independent cleanup attempts preserve the primary failure and partial evidence."""
import json
import traceback
from pathlib import Path

def write(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def initialize(obj,initializer,receipt):
    try:initializer(obj);return obj
    except BaseException as error:
        data=dict(primary_error=repr(error),primary_traceback=traceback.format_exc())
        try:data['stop_result']=obj.stop()
        except BaseException as cleanup:data['cleanup_error']=repr(cleanup)
        try:write(receipt,data)
        except BaseException as save_error:
            if hasattr(error,'add_note'):error.add_note('Constructor receipt failed: '+repr(save_error))
        raise

def cleanup_callbacks(callbacks):
    """Failure in one observation or save cannot bypass the remaining cleanup."""
    errors=[]
    for name,callback in callbacks:
        try:callback()
        except BaseException as error:errors.append(dict(stage=name,error=repr(error),traceback=traceback.format_exc()))
    return errors

def native_cleanup(record,scope,out):
    import numpy as np
    surface=scope.get('surface');cp=scope.get('cp_trial')
    def partial():
        # Preserve the entire available prefix on errors, including before the
        # final full trajectory export. Does not relabel a prefix as complete.
        if record.get('exit_code',1)==0:return
        arrays={k:np.asarray(scope[k]) for k in ('times','states','joints','targets','hand_states',
                'finger_states','qualities','native_positions','native_velocities') if k in scope}
        if arrays:np.savez_compressed(Path(out)/'partial-native-state.npz',**arrays)
    actions=[]
    if surface is not None and cp is not None:actions.append(('surface_snapshot',surface.snapshot))
    if cp is not None:actions.append(('cp_stop',lambda:record.update(cp_cleanup_errors=cp.stop())))
    actions.append(('partial_state',partial))
    errors=cleanup_callbacks(actions)
    if errors:
        record['cleanup_failures']=errors;record['exit_code']=1
    write(Path(out)/'native-finally-receipt.json',dict(primary_error=record.get('error'),
        exit_code=record.get('exit_code',1),cleanup_errors=errors,
        partial_state_present=(Path(out)/'partial-native-state.npz').exists(),
        cp_was_present=cp is not None))
