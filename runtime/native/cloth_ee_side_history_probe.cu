// Owned single-row interior EE candidate. Explicit caller-owned old geometry.
// Not a generator, CCD, general contact solver, or live-ABI integration.
#include <cuda_runtime.h>
#include <math.h>
struct SV{double x,y,z;};
__device__ SV sub(SV a,SV b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
__device__ SV mul(SV a,double b){return {a.x*b,a.y*b,a.z*b};}
__device__ SV add(SV a,SV b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
__device__ double dot(SV a,SV b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ SV cross(SV a,SV b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
__device__ SV xyz(float4 x){return {x.x,x.y,x.z};}
__device__ bool finite4(float4 x){return isfinite(x.x)&&isfinite(x.y)&&isfinite(x.z)&&isfinite(x.w)&&x.w>=0;}
__device__ bool geometry(const float4* p,SV& n,double& signed_gap,double& s,double& t){
 SV u=sub(xyz(p[1]),xyz(p[0])),v=sub(xyz(p[3]),xyz(p[2])),w=sub(xyz(p[0]),xyz(p[2]));n=cross(u,v);double nn=dot(n,n);
 if(!(dot(u,u)>1e-20&&dot(v,v)>1e-20&&nn>1e-10*dot(u,u)*dot(v,v)))return false;
 s=dot(cross(v,w),n)/nn;t=dot(cross(u,w),n)/nn;
 if(!(s>1e-6&&s<1-1e-6&&t>1e-6&&t<1-1e-6))return false;
 n=mul(n,1/sqrt(nn));signed_gap=dot(w,n);return isfinite(signed_gap);
}
extern "C" __global__ void cloth_ee_side_history_probe(const float4* previous,const float4* current,float4* result,float4* dv,double separation,double h,unsigned* status){
 if(blockIdx.x||threadIdx.x)return;
 *status=0;
 if(!(isfinite(separation)&&separation>=0&&isfinite(h)&&h>0)){*status=1;return;}
 for(int i=0;i<4;++i)if(!finite4(previous[i])||!finite4(current[i])){*status=2;return;}
 SV oldn,n;double oldgap,gap,os,ot,s,t;
 if(!geometry(previous,oldn,oldgap,os,ot)||!geometry(current,n,gap,s,t)||fabs(oldgap)<=1e-12){*status=3;return;}
 double side=oldgap>0?1:-1;gap*=side;n=mul(n,side);
 double weights[4]={1-s,s,-(1-t),-t},den=0;
 for(int i=0;i<4;++i)den+=current[i].w*weights[i]*weights[i];
 if(!(den>1e-20)){*status=4;return;}
 double lambda=fmax(0.,separation-gap)/den;
 for(int i=0;i<4;++i){SV delta=mul(n,lambda*current[i].w*weights[i]),p=xyz(current[i]);result[i]=make_float4(float(p.x+delta.x),float(p.y+delta.y),float(p.z+delta.z),current[i].w);
  SV actual=sub(xyz(result[i]),p);dv[i]=make_float4(float(actual.x/h),float(actual.y/h),float(actual.z/h),0);
 }
}
