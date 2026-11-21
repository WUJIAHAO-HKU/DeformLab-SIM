// Opt-in fixed-coupon gap correction. Default-off path adds no GPU launches.
namespace {
CUmodule gap_module=nullptr;
CUfunction gap_capture_fn=nullptr,gap_apply_fn=nullptr;
CUdeviceptr gap_cache=0,gap_state=0;
unsigned long long gap_frame=0,gap_prepares=0,gap_calls=0,gap_frame_calls=0;
bool gap_active(){return gap_module!=nullptr;}
void gap_free(){
 if(gap_cache)cuMemFree(gap_cache);if(gap_state)cuMemFree(gap_state);
 if(gap_module)cuModuleUnload(gap_module);
 gap_cache=gap_state=0;gap_module=nullptr;gap_capture_fn=gap_apply_fn=nullptr;
 gap_frame=gap_prepares=gap_calls=gap_frame_calls=0;
}
bool gap_complete(){return !gap_active() || (gap_calls==gap_prepares*(terminal_positions+1ull) &&
 (!gap_prepares || gap_frame_calls==terminal_positions+1ull));}
CUresult gap_prepare(CUstream stream){
 if(!gap_active())return CUDA_SUCCESS;
 unsigned n=unsigned(nodes),rounds=terminal_positions+1;
 unsigned long long frame=cp_frame;
 if(!root_residual_enabled || !residual || !sb || pc_policy!=4 || !stream || !frame ||
    (gap_prepares && (frame!=gap_frame+1 || gap_frame_calls!=rounds)))return CUDA_ERROR_NOT_SUPPORTED;
 void* args[]={&sb,&residual,&gap_cache,&n,&frame,&rounds,&gap_state,&rejected};
 CUresult c=real_launch()(gap_capture_fn,1,1,1,1,1,1,0,stream,args,nullptr);
 if(c==CUDA_SUCCESS){gap_frame=frame;++gap_prepares;gap_frame_calls=0;}
 return c;
}
CUresult gap_apply(CUstream stream,void** params){
 if(!gap_active())return CUDA_SUCCESS;
 unsigned n=unsigned(nodes),rounds=terminal_positions+1;double h=*static_cast<float*>(params[11]);
 unsigned long long frame=cp_frame;
 if(!gap_prepares || frame!=gap_frame || gap_frame_calls>=rounds)return CUDA_ERROR_NOT_SUPPORTED;
 void* args[]={&nr_rows,params[3],&gap_cache,&residual,&n,&h,&frame,&rounds,&gap_state,&rejected};
 CUresult c=real_launch()(gap_apply_fn,1,1,1,1,1,1,0,stream,args,nullptr);
 if(c==CUDA_SUCCESS){++gap_calls;++gap_frame_calls;}
 return c;
}
}
extern "C" int observer_contact_gap_start(const char* path){
 std::lock_guard<std::mutex> guard(lock);
 if(!path || !enabled || mode!=6 || !root_residual_enabled || !dc_active() || !nr_module ||
    !terminal_transport || !terminal_positions || pc_policy!=4 || gap_active() || gap_cache || gap_state ||
    sb || stream_seen || damping_calls || nr_calls || trace_file || angular_states)return -1;
 CUresult c=cuCtxPushCurrent(context);if(c!=CUDA_SUCCESS)return int(c);
 CUmodule next=nullptr;CUfunction capture=nullptr,apply=nullptr;CUdeviceptr cache=0,state=0;
 c=cuModuleLoad(&next,path);
 if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&capture,next,"capture_contact_gap_origin");
 if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&apply,next,"correct_contact_gap_indirect");
 if(c==CUDA_SUCCESS)c=cuMemAlloc(&cache,nodes*24);
 if(c==CUDA_SUCCESS)c=cuMemAlloc(&state,32);
 if(c==CUDA_SUCCESS)c=cuMemsetD8(state,0,32);
 if(c==CUDA_SUCCESS){gap_module=next;gap_capture_fn=capture;gap_apply_fn=apply;gap_cache=cache;gap_state=state;}
 else {if(state)cuMemFree(state);if(cache)cuMemFree(cache);if(next)cuModuleUnload(next);}
 CUcontext old;CUresult popped=cuCtxPopCurrent(&old);return int(c==CUDA_SUCCESS?popped:c);
}
extern "C" int observer_contact_gap_enabled(){return gap_active()?1:0;}
extern "C" unsigned long long observer_contact_gap_count(unsigned i){return i==0?gap_prepares:i==1?gap_calls:i==2?gap_frame:gap_frame_calls;}
