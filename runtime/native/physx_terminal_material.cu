// Identity through native CP copy chains, not a force or stiffness rescaling.
// The ordinary native launch has already updated its bookkeeping/rotation.
// Cached incoming coordinates and velocities are forwarded to the same remap
// destinations. Mass splitting and root merge remain on their original path.
#include <cuda_runtime.h>
template<class T> __device__ T field(char* sb,int offset){return *reinterpret_cast<T*>(sb+offset);}
extern "C" __global__ void transport_terminal_partition(char* sb,float4* cache,unsigned part){
    auto prefix=field<unsigned*>(sb,208);
    unsigned tet=(part?prefix[part-1]:0)+blockIdx.x*blockDim.x+threadIdx.x;
    if(tet>=prefix[part])return;
    uint4 remap=field<uint4*>(sb,192)[tet];
    auto cp=field<float4*>(sb,184);auto cv=field<float4*>(sb,176);
    for(int k=0;k<4;++k){cp[(&remap.x)[k]]=cache[8*tet+k];cv[(&remap.x)[k]]=cache[8*tet+4+k];}
}
