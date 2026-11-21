// Diagnostic-only single expression ablation in a matched native PTX clone.
namespace {
CUmodule friction_module=nullptr;CUfunction friction_function=nullptr;
int friction_policy=0;unsigned long long friction_calls=0;
void friction_free(){friction_policy=0;friction_function=nullptr;if(friction_module)cuModuleUnload(friction_module);friction_module=nullptr;}
}
extern "C" int observer_friction_trial_start(const char* path,int policy){
 if(!path||!enabled||mode!=9||rr_policy!=2||friction_policy||policy<1||policy>2)return -1;
 CUresult c=cuCtxPushCurrent(context);if(c!=CUDA_SUCCESS)return int(c);
 c=cuModuleLoad(&friction_module,path);
 if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&friction_function,friction_module,"sb_solveRigidSoftCollisionLaunchTGS");
 if(c==CUDA_SUCCESS){friction_policy=policy;friction_calls=0;}else friction_free();
 CUcontext previous;cuCtxPopCurrent(&previous);return int(c);
}
extern "C" unsigned long long observer_friction_trial_count(){return friction_calls;}
