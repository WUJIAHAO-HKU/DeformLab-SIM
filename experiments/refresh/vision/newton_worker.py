"""Isolated pinned Python/Warp runtime, synchronous physics batches only."""
from pathlib import Path
import argparse,json,os,sys,traceback
import numpy as np
import warp as wp

def main():
    p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--response-fd',type=int,required=True);p.add_argument('--out',required=True);p.add_argument('--root',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(exist_ok=True);response=os.fdopen(a.response_fd,'w',buffering=1)
    def send(x):response.write(json.dumps(x,allow_nan=False)+'\n');response.flush()
    wp.config.kernel_cache_dir=os.environ.get('WARP_CACHE_PATH','.cache/deformlabsim-diffusion180-v1/cache/warp')
    sys.path.insert(0,str(Path(a.root)/'work/paper-precision-gauge-v8/vision-v3'))
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from newton_graph_backend import Backend
    request=json.loads(Path(a.fixture).read_text())
    def save(samples,path):
        np.savez(path,**{k:np.asarray([v[k] for v in samples])for k in samples[0]})
    try:
        backend=Backend(request['fixture'],request['settings'])
        (out/'backend-receipt.json').write_text(json.dumps(backend.receipt,indent=2)+'\n')
        save([backend.sample()],out/'initial.npz');send(dict(status='ready',pid=os.getpid(),path=str(out/'initial.npz')))
        for line in sys.stdin:
            command=json.loads(line)
            if command['command']=='stop':
                (out/'compute-times.json').write_text(json.dumps(backend.compute_s)+'\n');send(dict(status='stopped',steps=backend.steps));return 0
            if command['command']!='advance':raise ValueError('Bad command')
            samples=backend.advance(command['correction'],command['offset'],command['count'])
            path=out/f'batch-{backend.steps:05d}.npz';save(samples,path)
            send(dict(status='advanced',step=backend.steps,path=str(path),compute_s=sum(backend.compute_s[-command['count']:])) )
    except BaseException as e:
        failure=dict(status='failed',error=repr(e),traceback=traceback.format_exc());(out/'failure.json').write_text(json.dumps(failure,indent=2)+'\n');send(failure);return 1
if __name__=='__main__':raise SystemExit(main())
