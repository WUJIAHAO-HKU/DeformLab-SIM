// Experimental matched-binary kernel-stage intervention, NOT an upstream patch.
// All corrections are local CP chain updates; no body momentum target is used.
#include <cuda_runtime.h>
template<class T> __device__ T field(char* sb,int offset){return *reinterpret_cast<T*>(sb+offset);}
extern "C" __global__ void split_root_mass(char* sb,float4* saved){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x,n=field<unsigned>(sb,372);if(i>=n)return;
    auto p=field<float4*>(sb,96);auto velocity=field<float4*>(sb,88);auto counts=field<unsigned*>(sb,200);
    float4 v=p[i];saved[i]=v;unsigned c=counts[i]-(i?counts[i-1]:0);
    v.w*=c;p[i]=v;
    // The constitutive denominator reads velocity.w, while applying its
    // positional correction reads position.w. Both must carry the same mass.
    float4 vel=velocity[i];vel.w=v.w;velocity[i]=vel;
}
extern "C" __global__ void prepare_average(char* sb,float4* saved,float4* scratch,int split){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x,n=field<unsigned>(sb,372);if(i>=n)return;
    auto p=field<float4*>(sb,96);auto v=field<float4*>(sb,88);auto d=field<float4*>(sb,128);
    float4 x=p[i];if(split){x.w=saved[i].w;p[i]=x;float4 vel=v[i];vel.w=x.w;v[i]=vel;}
    scratch[i]=x;scratch[n+i]=v[i];scratch[2*n+i]=d[i];
}
extern "C" __global__ void merge_copy_velocity(char* sb,float4* scratch,float invDt){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x,n=field<unsigned>(sb,372);if(i>=n)return;
    auto p=field<float4*>(sb,96);auto v=field<float4*>(sb,88);auto d=field<float4*>(sb,128);
    auto cv=field<float4*>(sb,176);auto counts=field<unsigned*>(sb,200);unsigned nt=field<unsigned>(sb,376);
    unsigned lo=i?counts[i-1]:0,hi=counts[i];float4 x=scratch[i],old=scratch[n+i],delta=scratch[2*n+i];
    double sx=0,sy=0,sz=0;
    for(unsigned j=lo;j<hi;++j){float4 a=cv[4*nt+j];sx+=double(a.x)-old.x;sy+=double(a.y)-old.y;sz+=double(a.z)-old.z;}
    double c=hi-lo;double dx=sx/c,dy=sy/c,dz=sz/c;
    v[i]=make_float4(float(double(old.x)+dx),float(double(old.y)+dy),float(double(old.z)+dz),x.w);
    dx/=invDt;dy/=invDt;dz/=invDt;
    p[i]=make_float4(float(double(x.x)+dx),float(double(x.y)+dy),float(double(x.z)+dz),x.w);
    d[i]=make_float4(float(double(delta.x)+dx),float(double(delta.y)+dy),float(double(delta.z)+dz),delta.w);
}

#include "physx_sequential_damping.cuh"

#include "physx_position_residual.cuh"

#include "physx_precision_state.cuh"
#include "physx_precision_cp.cuh"
#include "physx_contact_support.cuh"
#include "physx_rigid_reduction.cuh"
