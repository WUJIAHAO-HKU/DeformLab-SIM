// Read-only signed-volume observation. Working evidence buffers only.
#include <cuda_runtime.h>
#include <math.h>
extern "C" __global__ void cone_trace_check(const float4* p,const uint4* t,
 const double* reference,unsigned nt,unsigned long long* result){
 unsigned j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=nt)return;
 uint4 ids=t[j];float4 a=p[ids.x],b=p[ids.y],c=p[ids.z],d=p[ids.w];
 double ux=double(b.x)-a.x,uy=double(b.y)-a.y,uz=double(b.z)-a.z;
 double vx=double(c.x)-a.x,vy=double(c.y)-a.y,vz=double(c.z)-a.z;
 double wx=double(d.x)-a.x,wy=double(d.y)-a.y,wz=double(d.z)-a.z;
 double ratio=(ux*(vy*wz-vz*wy)+uy*(vz*wx-vx*wz)+uz*(vx*wy-vy*wx))/reference[j];
 if(!isfinite(ratio)||ratio<=0)atomicAdd(result+1,1ull);
 if(isfinite(ratio)){
  unsigned long long bits=__double_as_longlong(ratio);
  unsigned long long ordered=(bits>>63)?~bits:(bits^0x8000000000000000ull);
  atomicMin(result,ordered);
 }
}
