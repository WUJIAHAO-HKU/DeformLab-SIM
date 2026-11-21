#include "physx_rigid_reduction_state.h"
namespace {
int rr_policy=0;CUdeviceptr rr_state=0;CUfunction rr_capture=nullptr,rr_publish=nullptr;
unsigned long long rr_calls[2]={};
std::mutex rr_lock;CUdeviceptr rr_contact_count=0,rr_core=0,rr_shared=0;
CUresult rr_after_solve(CUstream stream,void** params){
 rr_contact_count=*static_cast<CUdeviceptr*>(params[3]);rr_core=*static_cast<CUdeviceptr*>(params[5]);rr_shared=*static_cast<CUdeviceptr*>(params[7]);
 void* p[]={params[1],params[3],params[4],params[5],params[7],params[8],params[10],&rr_state,&rejected,&rr_policy};
 auto c=real_launch()(rr_capture,1,1,1,64,1,1,0,stream,p,nullptr);if(c==CUDA_SUCCESS)++rr_calls[0];return c;
}
bool rr_matches(void** p){return p&&*static_cast<CUdeviceptr*>(p[1])==rr_contact_count&&*static_cast<CUdeviceptr*>(p[6])==rr_core&&*static_cast<CUdeviceptr*>(p[8])==rr_shared;}
CUresult rr_after_reduce(CUstream stream,void** params){
 void* p[]={params[6],params[8],&rr_state,&rejected,&rr_policy};
 auto c=real_launch()(rr_publish,1,1,1,64,1,1,0,stream,p,nullptr);if(c==CUDA_SUCCESS)++rr_calls[1];return c;
}
void rr_free(){rr_policy=0;if(rr_state)cuMemFree(rr_state);rr_state=0;}
}
extern "C" int observer_rigid_reduction_start(int policy){
 if(!enabled||mode!=9||rr_policy||policy<1||policy>2)return -1;
 CUresult c=cuCtxPushCurrent(context);if(c!=CUDA_SUCCESS)return int(c);
 c=cuModuleGetFunction(&rr_capture,module,"capture_rigid_reduction");
 if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&rr_publish,module,"publish_rigid_reduction");
 if(c==CUDA_SUCCESS)c=cuMemAlloc(&rr_state,64*sizeof(RigidReductionState));
 if(c==CUDA_SUCCESS)c=cuMemsetD8(rr_state,0,64*sizeof(RigidReductionState));
 CUcontext previous;cuCtxPopCurrent(&previous);if(c==CUDA_SUCCESS)rr_policy=policy;return int(c);
}
extern "C" unsigned long long observer_rigid_reduction_count(int kind){return kind>=0&&kind<2?rr_calls[kind]:0;}
extern "C" int observer_rigid_reduction_read(double* out){
 if(!out||!rr_policy)return -1;RigidReductionState rows[64];
 CUresult c=cuCtxPushCurrent(context);if(c!=CUDA_SUCCESS)return int(c);
 c=cuCtxSynchronize();if(c==CUDA_SUCCESS)c=cuMemcpyDtoH(rows,rr_state,sizeof(rows));CUcontext previous;cuCtxPopCurrent(&previous);if(c!=CUDA_SUCCESS)return int(c);
 for(int k=0;k<16;++k)out[k]=0;int found=0;
 for(auto s:rows)if(s.initialized){
  if(++found>1)return -2;
  out[0]=s.residual.x;out[1]=s.residual.y;out[2]=s.residual.z;
  out[3]=s.published.x;out[4]=s.published.y;out[5]=s.published.z;
  out[6]=s.native_roundoff_sum.x;out[7]=s.native_roundoff_sum.y;out[8]=s.native_roundoff_sum.z;
  out[9]=s.continuity_error.x;out[10]=s.continuity_error.y;out[11]=s.continuity_error.z;
  out[12]=s.samples;out[13]=s.faults;out[14]=s.pending;out[15]=1;
 }
 return 0;
}
