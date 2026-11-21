"""Explicit owned CUDA graphs; contact/query PTX is unchanged."""
import json
from pathlib import Path
from .cloth_stage_backend import sha
BACKEND='cloth-split-envelope-refresh16-edge-query-graph-contact'

def artifacts(root):
    from .cloth_edge_query_backend import artifacts as edge_artifacts
    root=Path(root);files=edge_artifacts(root);base_ptx=files['cloth_split_ptx']
    build=root/'results/cloth-owned-graph-backend-build-v1';m=json.loads((build/'build.json').read_text())
    files.update(library=build/'libcloth_split_backend.so',manifest=build/'build.json',cloth_split_ptx=build/'contact.ptx')
    if m['status']!='compiled' or not m.get('explicit_owned_graph_start_required') or not m.get('no_fallback') or sha(files['library'])!=m['library_sha256'] or sha(files['cloth_split_ptx'])!=m['ptx_sha256'] or sha(base_ptx)!=m['ptx_sha256']:raise ValueError('Verified identical-PTX graph build required')
    pins=dict(m['input_sha256'])
    for key,name in [('cloth_split_selftest','cloth-owned-graph-binding-v2'),('selftest','cloth-owned-graph-dispatch-v1'),('cache_selftest','cloth-owned-graph-cache-v1'),('edge_query_selftest','cloth-owned-graph-query-v1')]:
        f=root/'results'/name/'summary.json';r=json.loads(f.read_text())
        if not r['passed']:raise ValueError('Graph binding prerequisite failed')
        files[key]=f;pins.update(r['input_sha256'])
    if any(sha(p)!=h for p,h in pins.items()):raise ValueError('Graph build/test input changed')
    return files

def evidence(values,status,counts):
    v=list(map(int,values));expected=sum(counts.get(n,0) for n in ('cloth_solveClothClothDeltaVTLaunch','cloth_solveClothClothDeltaEELaunch'))
    if len(v)!=5:raise ValueError('Five graph counters required')
    valid=status==0 and expected>0 and v[0]>0 and v[1]>0 and v[0]+v[1]==v[2]==expected and v[3]==v[0] and v[4]==0
    return dict(builds=v[0],reuses=v[1],launches=v[2],destroyed=v[3],errors=v[4],exit_code=status,execution_verified=bool(valid),actual_contact_gpu_execution_required=True,performance_accepted=False,platform_acceptance=False)
