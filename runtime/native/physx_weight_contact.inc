// Explicit single-volume articulated weight experiment. Native material mode 3.
namespace {
CUmodule weight_module=nullptr;CUfunction weight_function=nullptr;
CUdeviceptr weight_statistics=0;unsigned long long weight_calls=0;
void weight_free(){
 weight_function=nullptr;
 if(weight_statistics)cuMemFree(weight_statistics);weight_statistics=0;
 if(weight_module)cuModuleUnload(weight_module);weight_module=nullptr;
}
}
extern "C" int observer_weight_contact_start(const char* path){
 if(!path||!*path||!enabled||mode!=3||weight_function||normal_policy||friction_policy||rr_policy||cp_policy||pc_policy)return -1;
 CUresult c=cuCtxPushCurrent(context);if(c!=CUDA_SUCCESS)return int(c);
 c=cuModuleLoad(&weight_module,path);
 if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&weight_function,weight_module,"sb_solveRigidSoftCollisionLaunchTGS");
 if(c==CUDA_SUCCESS)c=cuMemAlloc(&weight_statistics,32);
 if(c==CUDA_SUCCESS)c=cuMemsetD8(weight_statistics,0,32);
 if(c!=CUDA_SUCCESS)weight_free();else weight_calls=0;
 CUcontext previous;cuCtxPopCurrent(&previous);return int(c);
}
extern "C" int observer_weight_contact_stats(unsigned* output){
 if(!output||!weight_function||!enabled)return -1;
 CUresult c=cuCtxPushCurrent(context);if(c!=CUDA_SUCCESS)return int(c);
 c=cuCtxSynchronize();if(c==CUDA_SUCCESS)c=cuMemcpyDtoH(output,weight_statistics,32);
 CUcontext previous;cuCtxPopCurrent(&previous);return int(c);
}
extern "C" unsigned long long observer_weight_contact_count(){return weight_calls;}
