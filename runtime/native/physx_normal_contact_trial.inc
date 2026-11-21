// Opt-in matched-PTX contact ablation; one volume, frictionless mode-6 fixture.
namespace {
CUmodule normal_module=nullptr;CUfunction normal_function=nullptr;
int normal_policy=0;unsigned long long normal_calls=0;
void normal_free(){normal_policy=0;normal_function=nullptr;if(normal_module)cuModuleUnload(normal_module);normal_module=nullptr;}
}
extern "C" int observer_normal_contact_start(const char* path,int policy){
 if(!path||!enabled||mode!=6||normal_policy||friction_policy||rr_policy||policy<1||policy>2)return -1;
 CUresult c=cuCtxPushCurrent(context);if(c!=CUDA_SUCCESS)return int(c);
 c=cuModuleLoad(&normal_module,path);
 if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&normal_function,normal_module,"sb_solveRigidSoftCollisionLaunchTGS");
 if(c==CUDA_SUCCESS){normal_policy=policy;normal_calls=0;}else normal_free();
 CUcontext previous;cuCtxPopCurrent(&previous);return int(c);
}
extern "C" unsigned long long observer_normal_contact_count(){return normal_calls;}
