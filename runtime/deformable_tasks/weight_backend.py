"""Version-locked experimental robot contact weights; no implicit backend fallback."""
import hashlib,json
from pathlib import Path
SUPPORT_BACKENDS=('cp-contact-weights-query-control','cp-contact-weights-support')
WEIGHT_BACKENDS=('cp-contact-weights-control','cp-contact-weights')+SUPPORT_BACKENDS
def artifacts(root,backend):
    if backend not in WEIGHT_BACKENDS:raise ValueError('Explicit weight backend required')
    root=Path(root);support=backend in SUPPORT_BACKENDS
    build=root/('results/robot-support-backend-build-v1' if support else 'results/robot-weight-backend-build-v1');manifest=build/'build.json'
    r=json.loads(manifest.read_text())
    if r.get('status')!='compiled':raise ValueError('Compiled weight build required')
    expected=([build/'libphysx_support_hook.so',build/'normalized.ptx',build/'query-control.ptx',build/'query-positive.ptx'] if support else [build/'libphysx_weight_hook.so',build/'control.ptx',build/'normalized.ptx'])
    for p in expected:
        if r['artifacts_sha256'].get(str(p))!=hashlib.sha256(p.read_bytes()).hexdigest():raise ValueError('Weight artifact hash mismatch: '+str(p))
    for name,h in r['input_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=h:raise ValueError('Weight build input changed: '+name)
    if support:
        replay=root/'results/robot-contact-support-instrumented-replay-v1'
        verified=json.loads((replay/'summary.json').read_text())
        if not verified.get('passed') or len(verified.get('records',[]))!=256:raise ValueError('Instrumented query replay required')
        for name,h in verified['source_sha256'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=h:raise ValueError('Instrumented replay input changed: '+name)
        if expected[2].read_bytes()!=(replay/'control.ptx').read_bytes() or expected[3].read_bytes()!=(replay/'positive_support.ptx').read_bytes():raise ValueError('Query differs from verified replay')
        return dict(library=expected[0],ptx=expected[1],query_ptx=expected[2] if backend.endswith('-control') else expected[3],manifest=manifest)

    return dict(library=expected[0],ptx=expected[1] if backend.endswith('-control') else expected[2],manifest=manifest)
