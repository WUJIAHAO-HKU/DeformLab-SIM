// Explicit experimental terminal pose update, pinned Cartesian coupon only.
namespace {
CUmodule tp_module=nullptr;CUfunction tp_capture=nullptr,tp_apply=nullptr;
CUdeviceptr tp_saved=0;float tp_h=0.f;unsigned long long tp_frames=0;
void tp_free(){
 if(tp_saved)cuMemFree(tp_saved);tp_saved=0;
 if(tp_module)cuModuleUnload(tp_module);tp_module=nullptr;
 tp_capture=tp_apply=nullptr;tp_h=0.f;tp_frames=0;
}
CUresult tp_publish(Launch original,CUfunction f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned shared,CUstream stream,void** params,void** extra){
 auto fail=[](){++errors;return CUDA_ERROR_NOT_SUPPORTED;};
 if(!params||extra||!stream||!terminal_transport||!coupled_damping||tp_frames+1!=terminal_frames)return fail();
 float outer_dt=*static_cast<float*>(params[1]);
 if(!std::isfinite(outer_dt)||std::fabs(outer_dt-tp_h*terminal_positions)>2e-6f*outer_dt)return fail();
 CUstreamCaptureStatus capture;
 if(cuStreamIsCapturing(stream,&capture)!=CUDA_SUCCESS||capture!=CU_STREAM_CAPTURE_STATUS_NONE||cuStreamSynchronize(stream)!=CUDA_SUCCESS)return fail();
 unsigned char core[344],art[496];CUdeviceptr desc=*static_cast<CUdeviceptr*>(params[0]);
 if(cuMemcpyDtoH(core,desc,sizeof(core))!=CUDA_SUCCESS || field<unsigned>(core,44)!=1 ||
    field<unsigned>(core,248)!=4 || field<unsigned>(core,252)!=3)return fail();
 if(cuMemcpyDtoH(art,field<CUdeviceptr>(core,32),sizeof(art))!=CUDA_SUCCESS ||
    field<unsigned short>(art,16)!=4 || field<unsigned short>(art,18)!=3)return fail();
 CUdeviceptr dofs=field<CUdeviceptr>(core,184),links=field<CUdeviceptr>(core,136),public_q=field<CUdeviceptr>(art,104);
 if(!dofs||!links||!public_q||!tp_saved||!rejected)return fail();
 void* before[]={&dofs,&links,&tp_saved,&rejected};
 CUresult result=real_launch()(tp_capture,1,1,1,1,1,1,0,stream,before,nullptr);
 if(result==CUDA_SUCCESS)result=original(f,gx,gy,gz,bx,by,bz,shared,stream,params,nullptr);
 if(result==CUDA_SUCCESS){void* after[]={&dofs,&links,&public_q,&tp_saved,&tp_h,&rejected};
  result=real_launch()(tp_apply,1,1,1,1,1,1,0,stream,after,nullptr);}
 if(result!=CUDA_SUCCESS)++errors;else ++tp_frames;return result;
}
}
extern "C" int observer_terminal_pose_start(const char* path,float inner_h){
 if(!path||!enabled||mode!=6||!terminal_transport||!coupled_damping||tp_module||tp_saved||damping_calls||
    !std::isfinite(inner_h)||inner_h<=0.f)return -1;
 CUresult c=cuCtxPushCurrent(context);if(c!=CUDA_SUCCESS)return int(c);
 c=cuModuleLoad(&tp_module,path);
 if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&tp_capture,tp_module,"capture_terminal_velocities");
 if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&tp_apply,tp_module,"apply_terminal_pose");
 if(c==CUDA_SUCCESS)c=cuMemAlloc(&tp_saved,43*sizeof(float));
 if(c==CUDA_SUCCESS){tp_h=inner_h;tp_frames=0;}else tp_free();
 CUcontext previous;cuCtxPopCurrent(&previous);return int(c);
}
extern "C" unsigned long long observer_terminal_pose_count(){return tp_frames;}
