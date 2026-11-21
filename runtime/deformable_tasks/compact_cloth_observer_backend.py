"""Explicit compact-graph observation, never a production or timing fallback."""
import json
from pathlib import Path
from .cloth_stage_backend import sha
CONTROL='cloth-split-envelope-refresh16-compact-observe-control'
BACKEND='cloth-split-envelope-refresh16-edge-query-graph-compact-observe-contact'
BACKENDS=(CONTROL,BACKEND)
def artifacts(root):
 from .cloth_compact_grid_backend import artifacts as compact_artifacts
 root=Path(root);files=compact_artifacts(root);base_ptx=files['cloth_split_ptx'];build=root/'results/compact-cloth-observer-build-v1';m=json.loads((build/'build.json').read_text())
 if m.get('status')!='compiled' or m.get('contact_ptx_byte_identical') is not True or m.get('compact_dispatch_byte_identical') is not True or m.get('intrusive_stage_synchronization') is not True or m.get('graph_internal_nodes_observed') is not False:raise ValueError('Matching explicit compact observer required')
 files.update(library=build/'libcloth_split_backend.so',manifest=build/'build.json',cloth_split_ptx=build/'contact.ptx')
 if sha(files['library'])!=m['library_sha256'] or sha(files['cloth_split_ptx'])!=m['ptx_sha256'] or sha(base_ptx)!=m['ptx_sha256']:raise ValueError('Observer or deployed compact kernel changed')
 pins=dict(m['input_sha256'])
 expected={'cloth_split_selftest':('binding',1080),'cloth_stage_selftest':('capture',58),'edge_query_selftest':('query',195),'selftest':('dispatch',10),'cache_selftest':('cache',12)}
 for key,(name,count) in expected.items():
  path=root/('results/compact-cloth-observer-'+name+'-v1/summary.json');r=json.loads(path.read_text())
  if r.get('passed') is not True or len(r.get('checks',{}))!=count or not all(r['checks'].values()):raise ValueError('Compact observer prerequisite failed: '+name)
  files[key]=path;pins.update(r['input_sha256'])
 if any(sha(p)!=h for p,h in pins.items()):raise ValueError('Compact observer input changed')
 return files
