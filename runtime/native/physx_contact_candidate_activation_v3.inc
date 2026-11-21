// Actual Cartesian plane-contact path. The legacy CP policy is mutually exclusive.
namespace {
CUmodule ca_module=nullptr;CUfunction ca_function=nullptr;CUdeviceptr ca_state=0;
unsigned long long ca_calls=0,ca_native_queries=0,ca_stats[4]={};
bool ca_active(){return ca_module!=nullptr;}
void ca_free(){
 if(ca_state)cuMemFree(ca_state);if(ca_module)cuModuleUnload(ca_module);
 ca_state=0;ca_module=nullptr;ca_function=nullptr;ca_calls=ca_native_queries=0;std::memset(ca_stats,0,sizeof(ca_stats));
}
bool ca_complete(){return !ca_module || (!ca_stats[0] && ca_calls==ca_stats[1] && ca_calls==ca_native_queries);}
unsigned long long ca_start_guard(){
 unsigned long long mask=0;
 if(!enabled || mode!=6)mask|=1;
 if(!root_residual_enabled || !dc_active() || !nr_module)mask|=2;
 if(!terminal_transport || !terminal_positions || !gap_active())mask|=4;
 if(pc_policy!=4)mask|=8;
 if(cp_policy)mask|=16;
 if(ca_module || ca_state)mask|=32;
 if(sb || stream_seen)mask|=64;
 if(damping_calls || nr_calls || cp_queries || cp_prepares || pc_prepares)mask|=128;
 return mask;
}
CUresult ca_before_query(CUstream stream,void** params){
 if(!ca_module)return CUDA_SUCCESS;
 if(!params || !stream || !sb || !gap_active() || pc_policy!=4 || cp_policy || !rejected ||
    !stream_seen || stream!=solver_stream || nodes>4096)return CUDA_ERROR_NOT_SUPPORTED;
 CUstreamCaptureStatus capture;CUresult c=cuStreamIsCapturing(stream,&capture);
 if(c!=CUDA_SUCCESS)return c;
 if(capture!=CU_STREAM_CAPTURE_STATUS_NONE)return CUDA_ERROR_NOT_SUPPORTED;
 unsigned n=unsigned(nodes);
 void* args[]={params[0],params[1],params[2],params[3],&sb,&n,params[4],params[5],params[6],&ca_state,&rejected};
 c=real_launch()(ca_function,1,1,1,1,1,1,0,stream,args,nullptr);
 if(c==CUDA_SUCCESS)c=cuStreamSynchronize(stream);
 if(c==CUDA_SUCCESS)c=cuMemcpyDtoH(ca_stats,ca_state,sizeof(ca_stats));
 if(c!=CUDA_SUCCESS)return c;
 if(ca_stats[0] || ca_stats[1]!=ca_calls+1)return CUDA_ERROR_NOT_SUPPORTED;
 ++ca_calls;return CUDA_SUCCESS;
}
}
extern "C" unsigned long long observer_contact_candidates_start_guard(){return ca_start_guard();}
extern "C" int observer_contact_candidates_start(const char* path){
 std::lock_guard<std::mutex> guard(lock);
 if(!path || ca_start_guard())return -1;
 CUresult c=cuCtxPushCurrent(context);if(c!=CUDA_SUCCESS)return int(c);
 CUmodule next=nullptr;CUfunction fn=nullptr;CUdeviceptr state=0;
 c=cuModuleLoad(&next,path);
 if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&fn,next,"activate_generated_contact_candidates");
 if(c==CUDA_SUCCESS)c=cuMemAlloc(&state,sizeof(ca_stats));
 if(c==CUDA_SUCCESS)c=cuMemsetD8(state,0,sizeof(ca_stats));
 if(c==CUDA_SUCCESS){ca_module=next;ca_function=fn;ca_state=state;ca_calls=ca_native_queries=0;std::memset(ca_stats,0,sizeof(ca_stats));}
 else {if(state)cuMemFree(state);if(next)cuModuleUnload(next);}
 CUcontext old;CUresult popped=cuCtxPopCurrent(&old);return int(c==CUDA_SUCCESS?popped:c);
}
extern "C" int observer_contact_candidates_enabled(){return ca_active()?1:0;}
extern "C" unsigned long long observer_contact_candidates_count(unsigned i){return i<4?ca_stats[i]:ca_native_queries;}
