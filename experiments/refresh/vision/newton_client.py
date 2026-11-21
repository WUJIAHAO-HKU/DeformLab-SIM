"""Actual VBD states -> native RTX display, image-only controller -> VBD drives."""
from pathlib import Path
import json,os,select,subprocess,time
import numpy as np

class Client:
    def __init__(self,root,out,fixture,settings):
        self.out=Path(out)/'external';self.out.mkdir(exist_ok=False);root=Path(root)
        request=self.out/'fixture.json';request.write_text(json.dumps(dict(fixture=fixture,settings=settings),indent=2)+'\n')
        r,w=os.pipe();self.response=os.fdopen(r,'r');self.log=(self.out/'worker.log').open('w')
        env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','LD_','ISAAC','CARB','EXP_','DEFORMABLE','CUDA_PATH'))}
        env.update(PATH='/usr/local/bin:/usr/bin:/bin',PYTHONDONTWRITEBYTECODE='1',TMPDIR='/dev/shm',TEMP='/dev/shm',TMP='/dev/shm',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        self.process=subprocess.Popen([str(root/'.venv/bin/python'),'-B',str(Path(__file__).with_name('newton_worker.py')),'--root',str(root),'--fixture',str(request),'--out',str(self.out),'--response-fd',str(w)],
            env=env,stdin=subprocess.PIPE,stdout=self.log,stderr=subprocess.STDOUT,text=True,bufsize=1,pass_fds=(w,))
        os.close(w);self.ready=self.read(300);assert self.ready['status']=='ready',self.ready
        self.initial=self.load(self.ready['path']);self.events=[]
    def read(self,timeout):
        if not select.select([self.response],[],[],timeout)[0]:raise TimeoutError('VBD worker response')
        raw=self.response.readline()
        if not raw:raise RuntimeError('VBD worker closed; inspect worker.log')
        reply=json.loads(raw)
        if reply['status']=='failed':raise RuntimeError(reply)
        return reply
    def load(self,path):
        with np.load(path) as a:return {k:a[k].copy()for k in a.files}
    def advance(self,correction,offset,count):
        t=time.monotonic();self.process.stdin.write(json.dumps(dict(command='advance',correction=float(correction),offset=float(offset),count=int(count)))+'\n');self.process.stdin.flush()
        reply=self.read(300);assert reply['status']=='advanced';data=self.load(reply['path'])
        self.events.append(dict(step=reply['step'],batch_wall_s=time.monotonic()-t,compute_s=reply['compute_s']));return data
    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"command":"stop"}\n');self.process.stdin.flush();self.read(30);self.process.wait(timeout=30)
        self.log.close();self.response.close()
        (self.out/'client-events.json').write_text(json.dumps(self.events,indent=2)+'\n')
        if self.process.returncode!=0:raise RuntimeError('VBD worker exit '+str(self.process.returncode))

def publish(runtime,gauge_views,data):
    """Render-only publication, no write to either backend's physical state."""
    b=runtime.bodies[0];p=np.asarray(data['positions'][-1],np.float32);b['last']=p.copy()
    b['visual'].GetPointsAttr().Set(runtime.Vt.Vec3fArray.FromNumpy(p))
    b['visual'].GetExtentAttr().Set(runtime.Vt.Vec3fArray.FromNumpy(np.asarray([p.min(0),p.max(0)],np.float32)))
    faces=b['faces'];normals=np.zeros_like(p);fn=np.cross(p[faces[:,1]]-p[faces[:,0]],p[faces[:,2]]-p[faces[:,0]])
    for j in range(3):np.add.at(normals,faces[:,j],fn)
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    b['visual'].GetNormalsAttr().Set(runtime.Vt.Vec3fArray.FromNumpy(normals));b['visual'].SetNormalsInterpolation('vertex')
    from pxr import UsdGeom,Gf
    for prim,pose in [(runtime.mover_visuals[0],data['tool'][-1]),(gauge_views[2],data['gauge'][-1])]:
        prim.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*[float(v)for v in pose[:3]]))
        attr=prim.GetAttribute('xformOp:orient')
        if not attr.IsValid():attr=UsdGeom.Xformable(prim).AddOrientOp().GetAttr()
        if str(attr.GetTypeName())=='quatd':q=Gf.Quatd(float(pose[6]),Gf.Vec3d(*[float(v)for v in pose[3:6]]))
        else:q=Gf.Quatf(float(pose[6]),Gf.Vec3f(*[float(v)for v in pose[3:6]]))
        attr.Set(q)
    # Assert actual display nodes, not just commanded positions.
    return float(np.max(np.abs(np.asarray(b['visual'].GetPointsAttr().Get())-p)))

class GaugeRecorder:
    def __init__(self,spec,hz):self.spec=spec;self.hz=hz;self.times=[];self.commands=[];self.actual=[];self.velocities=[]
    def append(self,data):
        from gauge_articulation import target
        for i,step in enumerate(data['step']):
            t=float(step)/self.hz;z,_=target(t,self.spec)
            self.times.append(t);self.commands.append([0.,0.,self.spec['initial_center_z_m']+z,0.,0.,0.,1.]);self.actual.append(data['gauge'][i]);self.velocities.append(data['gauge_velocity'][i])
    def save(self,path):
        if not self.times:return dict(samples=0,partial=True)
        np.savez_compressed(path,time_s=self.times,commanded_pose_xyzw=self.commands,actual_pose_xyzw=self.actual,actual_velocity=self.velocities)
        return dict(samples=len(self.times),source='Newton GPU solved body poses, every step',
            publication='Newton VBD articulated dynamics; no rollout pose setters',
            maximum_pose_error_m=float(np.max(abs(np.asarray(self.actual)[:,:3]-np.asarray(self.commands)[:,:3]))),reaction_force_observed=False)
