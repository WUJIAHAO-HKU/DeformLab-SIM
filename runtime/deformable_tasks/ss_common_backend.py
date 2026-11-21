"""Explicit two-volume common-relaxation experiments; no surface/robot fallback."""
import hashlib,json
from pathlib import Path
COMMON_BACKENDS={'ss-common-control':0,'ss-common-history':1,'ss-common-nodes':2,'ss-cone-control':1,'ss-cone-plane':1,'ss-cp-observe-native':0,'ss-cp-map-native':0,'ss-cp-map-common':1,'volume-mapping':0}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while data:=f.read(4*1024*1024):h.update(data)
    return h.hexdigest()
def artifacts(root):
    root=Path(root);build=root/'results/multisoft-common-binding-build-v1'
    files=dict(library=build/'libmultisoft_common_binding.so',manifest=build/'build.json',
        selftest=root/'results/multisoft-common-binding-selftest-v1/summary.json',
        dispatch_selftest=root/'results/multisoft-common-binding-dispatch-v1/summary.json',
        ptx=root/'results/multisoft-common-history-owned-v1/common-history.ptx',
        common_owned=root/'results/multisoft-common-history-owned-v1/summary.json',
        common_negative=root/'results/multisoft-common-history-invalid-v1/summary.json')
    manifest=json.loads(files['manifest'].read_text())
    if manifest.get('status')!='compiled' or sha(files['library'])!=manifest['library_sha256']:raise ValueError('Compiled common binding required')
    for key in ('manifest','selftest','dispatch_selftest','common_negative'):
        record=json.loads(files[key].read_text())
        if key!='manifest' and not record.get('passed'):raise ValueError('Common binding self-test failed')
        for p,h in record['input_sha256'].items():
            if sha(p)!=h:raise ValueError('Common binding input changed: '+p)
    owned=json.loads(files['common_owned'].read_text())
    if not owned['passed'] or sha(files['ptx'])!=owned['input_sha256'][str(files['ptx'])]:raise ValueError('Verified history kernel required')
    return files
