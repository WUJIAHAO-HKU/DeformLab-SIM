// Restricted CP mapping experiment. Constitutive kernels remain native.
#include <cuda_runtime.h>
extern "C" __global__ void mm_split(float4* p,float4* v,const unsigned* ends,unsigned n,float4* saved){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 float4 x=p[i];saved[i]=x;unsigned count=ends[i]-(i?ends[i-1]:0);
 x.w*=count;p[i]=x;float4 vel=v[i];vel.w=x.w;v[i]=vel;
}
extern "C" __global__ void mm_prepare(float4* p,float4* v,const float4* d,unsigned n,const float4* saved,float4* scratch){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 float4 x=p[i],vel=v[i];x.w=saved[i].w;vel.w=x.w;p[i]=x;v[i]=vel;
 scratch[i]=x;scratch[n+i]=vel;scratch[2*n+i]=d[i];
}
extern "C" __global__ void mm_merge(float4* p,float4* v,float4* d,const float4* cv,const unsigned* ends,unsigned n,unsigned nt,const float4* scratch,float invdt){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 unsigned lo=i?ends[i-1]:0,hi=ends[i];float4 old=scratch[n+i],x=scratch[i],delta=scratch[2*n+i];
 double dx=0,dy=0,dz=0;
 for(unsigned j=lo;j<hi;++j){float4 a=cv[4*nt+j];dx+=double(a.x)-old.x;dy+=double(a.y)-old.y;dz+=double(a.z)-old.z;}
 double c=hi-lo;dx/=c;dy/=c;dz/=c;
 v[i]=make_float4(float(double(old.x)+dx),float(double(old.y)+dy),float(double(old.z)+dz),x.w);
 dx/=invdt;dy/=invdt;dz/=invdt;
 p[i]=make_float4(float(double(x.x)+dx),float(double(x.y)+dy),float(double(x.z)+dz),x.w);
 d[i]=make_float4(float(double(delta.x)+dx),float(double(delta.y)+dy),float(double(delta.z)+dz),delta.w);
}
