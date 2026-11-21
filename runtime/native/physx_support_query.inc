// Fused positive-weight reference counts, paired with the weight experiment.
namespace {
CUmodule support_module=nullptr;CUfunction support_function=nullptr;
CUdeviceptr support_statistics=0;unsigned long long support_calls=0;
void support_free(){
 support_function=nullptr;
 if(support_statistics)cuMemFree(support_statistics);support_statistics=0;
 if(support_module)cuModuleUnload(support_module);support_module=nullptr;
}
}
extern "C" int observer_support_query_start(const char* path){
 if(!path||!*path||!enabled||mode!=3||!weight_function||support_function)return -1;
 CUresult c=cuCtxPushCurrent(context);if(c!=CUDA_SUCCESS)return int(c);
 c=cuModuleLoad(&support_module,path);
 if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&support_function,support_module,"sb_queryRigidSoftContactReferenceCountLaunchTGS");
 if(c==CUDA_SUCCESS)c=cuMemAlloc(&support_statistics,8);
 if(c==CUDA_SUCCESS)c=cuMemsetD8(support_statistics,0,8);
 if(c!=CUDA_SUCCESS)support_free();else support_calls=0;
 CUcontext previous;cuCtxPopCurrent(&previous);return int(c);
}
extern "C" int observer_support_query_stats(unsigned* output){
 if(!output||!support_function||!enabled)return -1;
 CUresult c=cuCtxPushCurrent(context);if(c!=CUDA_SUCCESS)return int(c);
 c=cuCtxSynchronize();if(c==CUDA_SUCCESS)c=cuMemcpyDtoH(output,support_statistics,8);
 CUcontext previous;cuCtxPopCurrent(&previous);return int(c);
}
extern "C" unsigned long long observer_support_query_count(){return support_calls;}
