"""A separate strict 13.5-second task schema; the frozen robot schema is untouched."""
import copy,hashlib,json
from pathlib import Path
from deformable_tasks import robot_config as base

def default_robot_config():
    c=base.default_robot_config();c.update(schema_version=2,task='robot_visual_regrasp',duration_s=13.5)
    return c
def base_config(config):
    if config.get('schema_version')!=2 or config.get('task')!='robot_visual_regrasp' or config.get('duration_s')!=13.5:
        raise ValueError('Expected explicit 13.5-second visual-regrasp schema v2')
    c=copy.deepcopy(config);c.update(schema_version=1,task='robot_volume',duration_s=7.5)
    return base.validate_robot_config(c)
def load_robot_config(path):
    c=json.loads(Path(path).read_text());base_config(c);return c
def robot_plan(config,backend):
    c,meta=base.robot_plan(base_config(config),backend)
    c.update(schema_version=2,task='robot_visual_regrasp',duration_s=13.5)
    meta.update(task='robot_visual_regrasp',config_sha256=hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        task_scope='Separate release/reacquire/regrasp entry; unchanged original 7.5-second robot schema')
    return c,meta
def recorder_arguments(path,config,backend,render=False):
    return base.recorder_arguments(path,base_config(config),backend,render)
