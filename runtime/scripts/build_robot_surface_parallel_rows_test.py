"""Expose the parallel plan under the original primitive ABI for oracle tests."""
import json,shutil,subprocess
from pathlib import Path
from audit_robot_contact_multiplier_lifecycle import sha
ROOT=Path(__file__).resolve().parents[1]
def main():
 out=ROOT/'results/robot-surface-parallel-rows-test-build-v1'
 if out.exists():raise ValueError('Fresh build required')
 out.mkdir();files=[Path(__file__),*[ROOT/'native'/p for p in ('robot_surface_stream_state.h','robot_table_surface_persistent_rows.cu','robot_surface_parallel_plan.inc')]]
 for p in files[1:]:shutil.copy2(p,out/p.name)
 original=(out/'robot_table_surface_persistent_rows.cu').read_text();start=original.index('extern "C" __global__ void robot_surface_plan(');end=original.index('{',start)
 text='#include "robot_surface_stream_state.h"\n#define robot_surface_plan serial_robot_surface_plan\n#include "robot_table_surface_persistent_rows.cu"\n#undef robot_surface_plan\ntemplate<class T> static __device__ T* ptr(uint64_t v){return reinterpret_cast<T*>(v);}\n#include "robot_surface_parallel_plan.inc"\n'+original[start:end]+'{\n SurfaceStreamState s={};\n'
 for field,arg in [('input[0]','info'),('input[1]','solver_ids'),('input[2]','constraints'),('input[4]','count'),('input[5]','tets'),('input[6]','position'),('input[7]','delta'),('boundary','boundary'),('plan','plan'),('selected','selected'),('cache','saved_lambda'),('seen','seen'),('header','header')]:text+=f' s.{field}=(uint64_t){arg};\n'
 for field,arg in [('nodes','nodes'),('tets','ntets'),('boundary_count','boundary_count'),('capacity','capacity'),('cx','center_x'),('cy','center_y'),('hx','half_x'),('hy','half_y'),('plane','plane'),('epsilon','epsilon')]:text+=f' s.{field}={arg};\n'
 text+=' parallel_surface_plan(&s,frame,round);\n}\n';(out/'wrapper.cu').write_text(text)
 command=['/usr/local/cuda-12.6/bin/nvcc','-ptx','-arch=sm_89','-O3',str(out/'wrapper.cu'),'-o',str(out/'rows.ptx')];r=subprocess.run(command,capture_output=True,text=True);(out/'build.log').write_text(r.stdout+r.stderr)
 result=dict(exit_code=r.returncode,input_sha256={str(p):sha(p) for p in files},wrapper_sha256=sha(out/'wrapper.cu'),ptx_sha256=sha(out/'rows.ptx') if (out/'rows.ptx').exists() else None,command=command,platform_acceptance=False);(out/'build.json').write_text(json.dumps(result,indent=2)+'\n');r.check_returncode()
if __name__=='__main__':main()
