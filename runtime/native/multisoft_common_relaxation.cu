// Experimental common relaxation for exactly two already accumulated SS buffers.
// No live-native binding here. Native applyExternalDeltas performs x/v publication.
#include <cuda_runtime.h>
#include <math.h>
extern "C" __global__ void multisoft_common_relaxation(
    float4* a,unsigned na,float4* b,unsigned nb,int mode,unsigned long long* stats) {
  __shared__ unsigned degree;
  __shared__ unsigned invalid;
  if(threadIdx.x==0){degree=1;invalid=(gridDim.x!=1||blockDim.x!=128||!a||!b||a==b||!na||!nb||na>100000||nb>100000||(mode!=0&&mode!=1));}
  __syncthreads();
  if(invalid){if(threadIdx.x==0){atomicAdd(stats,1ull);atomicAdd(stats+1,1ull);}return;}
  for(unsigned i=threadIdx.x;i<na+nb;i+=blockDim.x){
    float4 c=i<na?a[i]:b[i-na];
    bool bad=!isfinite(c.x)||!isfinite(c.y)||!isfinite(c.z)||!isfinite(c.w)||c.w<0||c.w>16384||c.w!=floorf(c.w)||(c.w==0&&(c.x!=0||c.y!=0||c.z!=0));
    if(bad)atomicOr(&invalid,1u);else atomicMax(&degree,(unsigned)c.w);
  }
  __syncthreads();
  if(threadIdx.x==0){atomicAdd(stats,1ull);if(invalid)atomicAdd(stats+1,1ull);else atomicMax(stats+4,(unsigned long long)degree);}
  if(invalid)return; // Whole-group validation precedes every in-place write.
  for(unsigned i=threadIdx.x;i<na+nb;i+=blockDim.x){
    float4* p=i<na?a+i:b+i-na;float4 c=*p;
    if(c.w>0){atomicAdd(stats+2,1ull);if(mode==1&&c.w!=(float)degree){p->w=(float)degree;atomicAdd(stats+3,1ull);}}
  }
}
