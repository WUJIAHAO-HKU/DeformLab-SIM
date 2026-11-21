"""Freeze and compile stream-ordered row bridge without changing live routes."""
import hashlib,json,re,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=ROOT/'results/robot-surface-parallel-runtime-build-v2'
 if out.exists():raise ValueError('Fresh build required')
 original=ROOT/'native/robot_table_surface_persistent_rows.cu'
 if sha(original)!='cb8f56593b4bdc83feea6b5e0a073d8f3a71b66b5266224f850f013c8bb8b83d':raise ValueError('Row arithmetic changed')
 proof=ROOT/'results/robot-surface-runtime-v1/summary.json'
 if not json.loads(proof.read_text())['passed']:raise ValueError('Original bridge prerequisite failed')
 out.mkdir();src=out/'source';src.mkdir()
 files=[original,Path(__file__),proof,*[ROOT/'native'/k for k in ('robot_surface_stream_state.h','robot_surface_parallel_rows.cu','robot_surface_stream_runtime.cpp','robot_surface_parallel_plan.inc')]]
 for p in files:
  if p.suffix in ('.h','.cu','.cpp','.inc'):shutil.copy2(p,src/p.name)
 text=original.read_text();text,n=re.subn(r'extern "C" __global__ void (robot_surface_\w+)',r'static __device__ void device_\1',text)
 if n!=6:raise ValueError('Unexpected primitive set')
 for name in ('append','history_restore'):
  start=text.index('static __device__ void device_robot_surface_'+name+'(');end=text.find('\n}',start)+2;part=text[start:end]
  if part.count('){')!=1 or part.count('j=blockIdx.x*blockDim.x+threadIdx.x')!=1:raise ValueError('Chunk conversion is not unique')
  part=part.replace('){',',unsigned base){').replace('j=blockIdx.x*blockDim.x+threadIdx.x','j=base+blockIdx.x*blockDim.x+threadIdx.x')
  text=text[:start]+part+text[end:]
 (src/'robot_surface_device_rows.inc').write_text(text)
 commands=[['/usr/local/cuda-12.6/bin/nvcc','-ptx','-arch=sm_89','-O3',str(src/'robot_surface_parallel_rows.cu'),'-o',str(out/'rows.ptx')],['g++','-std=c++17','-shared','-fPIC','-O2','-Wall','-Wextra','-I/usr/local/cuda-12.6/include',str(src/'robot_surface_stream_runtime.cpp'),'-lcuda','-o',str(out/'librobot_surface_runtime.so')]]
 result=dict(input_sha256={str(p):sha(p) for p in files},source_sha256={p.name:sha(p) for p in src.iterdir()},commands=commands,platform_registered=False,platform_acceptance=False)
 logs=[];code=0
 for cmd in commands:
  r=subprocess.run(cmd,capture_output=True,text=True);logs.append(r.stdout+r.stderr);code=r.returncode
  if code:break
 (out/'build.log').write_text('\n'.join(logs));result.update(exit_code=code,library_sha256=sha(out/'librobot_surface_runtime.so') if (out/'librobot_surface_runtime.so').exists() else None,ptx_sha256=sha(out/'rows.ptx') if (out/'rows.ptx').exists() else None)
 (out/'build.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(exit_code=code,library_sha256=result['library_sha256'])))
 if code:raise RuntimeError('Build failed; see frozen build.log')
if __name__=='__main__':main()
