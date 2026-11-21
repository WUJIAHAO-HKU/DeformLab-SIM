"""Explicit compact launch grids, with unchanged repaired contact PTX."""
import json
from pathlib import Path
from .cloth_stage_backend import sha
BACKEND='cloth-split-envelope-refresh16-edge-query-graph-compact-contact'
def artifacts(root):
 from .cloth_reduced_count_backend import artifacts as reduced_artifacts
 root=Path(root);files=reduced_artifacts(root);previous_ptx=files['cloth_split_ptx'];previous_manifest=files['manifest']
 build=root/'results/cloth-compact-grid-graph-build-v1';m=json.loads((build/'build.json').read_text())
 if m.get('status')!='compiled' or m.get('contact_grid_cap')!=128 or not m.get('all_rows_via_unchanged_grid_stride_loops') or not m.get('contact_ptx_byte_identical'):raise ValueError('Verified unchanged-PTX compact grid required')
 files.update(library=build/'libcloth_split_backend.so',manifest=build/'build.json',cloth_split_ptx=build/'contact.ptx',native_binding_manifest=previous_manifest)
 if sha(files['library'])!=m['library_sha256'] or sha(files['cloth_split_ptx'])!=m['ptx_sha256'] or sha(previous_ptx)!=m['ptx_sha256'] or m['binary_sha256']!=json.loads(previous_manifest.read_text())['binary_sha256']:raise ValueError('Compact artifact/native ABI changed')
 pins=dict(m['input_sha256'])
 for key,name,expected in [('cloth_split_selftest','binding',960),('selftest','dispatch',10),('cache_selftest','cache',12),('edge_query_selftest','query',131)]:
  path=root/('results/cloth-compact-grid-'+name+'-v1/summary.json');r=json.loads(path.read_text())
  if r.get('passed') is not True or len(r.get('checks',{}))!=expected or not all(r['checks'].values()):raise ValueError('Compact-grid prerequisite failed')
  files[key]=path;pins.update(r['input_sha256'])
 if any(sha(p)!=h for p,h in pins.items()):raise ValueError('Compact-grid build/test input changed')
 return files
