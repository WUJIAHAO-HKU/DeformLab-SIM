// Opt-in contact/drive coupling; numerical source and binaries are isolated by builder.
namespace {
CUmodule dc_module=nullptr,dc_drive_module=nullptr;
CUfunction dc_prepare_fn=nullptr,dc_reduce_fn=nullptr,dc_apply_fn=nullptr;
CUdeviceptr dc_state=0,dc_delta=0;
alignas(8) unsigned long long dc_values[15]={};
unsigned dc_grid[3]={},dc_block[3]={},dc_shared=0;
CUstream dc_stream=nullptr;
bool dc_have_drive=false,dc_pending=false;
unsigned long long dc_drives=0,dc_contacts=0,dc_applies=0;
const size_t dc_sizes[15]={8,4,4,1,4,4,4,8,8,8,1,1,1,4,1};
bool dc_active(){return dc_module!=nullptr;}
void dc_free(){
 for(auto p:{dc_state,dc_delta})if(p)cuMemFree(p);
 dc_state=dc_delta=0;
 if(dc_module)cuModuleUnload(dc_module);if(dc_drive_module)cuModuleUnload(dc_drive_module);
 dc_module=dc_drive_module=nullptr;dc_prepare_fn=dc_reduce_fn=dc_apply_fn=nullptr;
 dc_stream=nullptr;dc_have_drive=dc_pending=false;dc_drives=dc_contacts=dc_applies=0;
 std::memset(dc_values,0,sizeof(dc_values));
}
bool dc_bind(void** params,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned shared,CUstream stream){
 if(!params||!stream||gx!=1||gy!=1||gz!=1||bx!=32||by!=1||bz!=1||shared||dc_pending||dc_have_drive)return false;
 for(unsigned i=0;i<15;++i)if(!params[i])return false;
 float h=*static_cast<float*>(params[1]);
 if(!std::isfinite(h)||h<=0.f||std::fabs(h-tp_h)>2e-6f*tp_h ||
    *static_cast<unsigned char*>(params[3])>1 || *static_cast<unsigned*>(params[13])!=1)return false;
 for(unsigned i:{10u,11u,12u,14u})if(*static_cast<unsigned char*>(params[i])!=1)return false;
 CUstreamCaptureStatus captured;if(cuStreamIsCapturing(stream,&captured)!=CUDA_SUCCESS||captured!=CU_STREAM_CAPTURE_STATUS_NONE)return false;
 for(unsigned i=0;i<15;++i){dc_values[i]=0;std::memcpy(&dc_values[i],params[i],dc_sizes[i]);}
 dc_grid[0]=gx;dc_grid[1]=gy;dc_grid[2]=gz;dc_block[0]=bx;dc_block[1]=by;dc_block[2]=bz;
 dc_shared=shared;dc_stream=stream;dc_have_drive=true;++dc_drives;return true;
}
CUresult dc_prepare(CUstream stream,void** params){
 if(!dc_have_drive||dc_pending||stream!=dc_stream||!params ||
    *static_cast<CUdeviceptr*>(params[6])!=dc_values[0])return CUDA_ERROR_NOT_SUPPORTED;
 float h=*static_cast<float*>(params[11]);float saved_h;std::memcpy(&saved_h,&dc_values[1],4);
 if(h!=saved_h)return CUDA_ERROR_NOT_SUPPORTED;
 unsigned phase=static_cast<unsigned char>(dc_values[3]);
 void* args[]={params[6],params[11],&phase,&nr_inputs,&dc_state,&dc_delta,&rejected};
 return real_launch()(dc_prepare_fn,1,1,1,1,1,1,0,stream,args,nullptr);
}
CUresult dc_reduce(CUstream stream,void** params){
 void* args[]={&nr_rows,params[3],&nr_target,&dc_state,&dc_delta,&rejected};
 return real_launch()(dc_reduce_fn,1,1,1,1,1,1,0,stream,args,nullptr);
}
void dc_after_contact(){dc_pending=true;++dc_contacts;}
CUresult dc_push(Launch original,CUfunction f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned shared,CUstream stream,void** params,void** extra){
 if(!dc_pending || !dc_have_drive || stream!=dc_stream || !params || extra)return CUDA_ERROR_NOT_SUPPORTED;
 CUresult c=original(f,gx,gy,gz,bx,by,bz,shared,stream,params,extra);
 if(c==CUDA_SUCCESS && !dc_observe("before"))return CUDA_ERROR_NOT_SUPPORTED;
 if(c==CUDA_SUCCESS){
  unsigned mode=2;void* args[17];for(unsigned i=0;i<15;++i)args[i]=&dc_values[i];args[15]=&mode;args[16]=&dc_delta;
  c=real_launch()(dc_apply_fn,dc_grid[0],dc_grid[1],dc_grid[2],dc_block[0],dc_block[1],dc_block[2],dc_shared,stream,args,nullptr);
 }
 if(c==CUDA_SUCCESS && !dc_observe("after"))return CUDA_ERROR_NOT_SUPPORTED;
 if(c==CUDA_SUCCESS){dc_pending=dc_have_drive=false;++dc_applies;}
 return c;
}
}
extern "C" int observer_drive_contact_start(const char* stages,const char* transport){
 if(!stages||!transport||!enabled||mode!=6||!nr_module||!tp_module||!terminal_transport||!coupled_damping||
    damping_calls||nr_calls||dc_module||dc_drive_module||dc_state||dc_delta)return -1;
 CUresult c=cuCtxPushCurrent(context);if(c!=CUDA_SUCCESS)return int(c);
 c=cuModuleLoad(&dc_module,stages);
 if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&dc_prepare_fn,dc_module,"prepare_drive_contact");
 if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&dc_reduce_fn,dc_module,"reduce_contact_drive_increment");
 if(c==CUDA_SUCCESS)c=cuModuleLoad(&dc_drive_module,transport);
 if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&dc_apply_fn,dc_drive_module,"artiSolveInternalConstraintsTGS1T");
 if(c==CUDA_SUCCESS)c=cuMemAlloc(&dc_state,8*sizeof(double));
 if(c==CUDA_SUCCESS)c=cuMemAlloc(&dc_delta,3*sizeof(float));
 if(c!=CUDA_SUCCESS)dc_free();
 CUcontext previous;cuCtxPopCurrent(&previous);return int(c);
}
extern "C" unsigned long long observer_drive_contact_count(unsigned kind){return kind==0?dc_drives:kind==1?dc_contacts:kind==2?dc_applies:0;}
