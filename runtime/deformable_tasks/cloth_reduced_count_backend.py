"""Strict selection of the counter-aggregated, physically equivalent graph module."""
import json
from pathlib import Path
from .cloth_stage_backend import sha
BACKEND='cloth-split-envelope-refresh16-edge-query-graph-reduced-contact'

def artifacts(root):
    from .cloth_owned_graph_backend import artifacts as graph_artifacts
    root=Path(root);files=graph_artifacts(root);original_library=files['library']
    build=root/'results/cloth-reduced-count-module-v1';m=json.loads((build/'build.json').read_text())
    # The unchanged host binding's manifest pins the native GPU ABI. Keep it
    # as the runtime manifest; the counter-only module has separate provenance.
    files.update(library=build/'libcloth_split_backend.so',reduced_count_manifest=build/'build.json',cloth_split_ptx=build/'contact.ptx')
    if m['status']!='compiled' or m.get('counter_offsets_aggregated')!=[4,5,7] or m.get('physical_source_replacements') is not False or m.get('host_library_byte_identical') is not True:raise ValueError('Counter-only module required')
    if sha(files['library'])!=m['library_sha256'] or sha(original_library)!=m['library_sha256'] or sha(files['cloth_split_ptx'])!=m['ptx_sha256']:raise ValueError('Counter module artifact changed')
    receipt=root/'results/cloth-reduced-count-binding-v1/summary.json';r=json.loads(receipt.read_text())
    if r.get('passed') is not True or len(r['checks'])!=960 or not all(r['checks'].values()):raise ValueError('Counter and physical equivalence required')
    pins={**m['input_sha256'],**r['input_sha256']}
    if any(sha(p)!=h for p,h in pins.items()):raise ValueError('Counter build/reference changed')
    files['reduced_count_selftest']=receipt;files['cloth_split_selftest']=receipt
    return files
