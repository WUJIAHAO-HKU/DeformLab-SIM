// Included inside the pinned census namespace; single-cloth, explicit opt-in only.
static bool cf_enabled=false,cf_pending=false;
static int cf_mode=0;static unsigned cf_n=0,cf_errors=0;
static unsigned long long cf_steps=0,cf_solves[2]{},cf_publications=0;
static CUcontext cf_context=nullptr;static CUmodule cf_module=nullptr;
static CUdeviceptr cf_table=0,cf_ids=0,cf_p=0,cf_v=0,cf_delta=0;
static CUdeviceptr cf_normal_table=0,cf_friction_table=0,cf_positions=0,cf_degree=0,cf_counts=0;
static CUfunction cf_prepare=nullptr,cf_bind_frozen=nullptr,cf_restore=nullptr,cf_publish=nullptr,cf_normal[2]{},cf_friction[2]{},cf_query[2]{};
static std::string cf_ptx;static float cf_h=0;static CUstream cf_stream=nullptr;
static CUresult cf_fail(){++cf_errors;return CUDA_ERROR_INVALID_VALUE;}
template<class T> static T cf_get(const unsigned char* p,unsigned off){T v;std::memcpy(&v,p+off,sizeof(v));return v;}
static bool cf_read(void* dst,CUdeviceptr src,size_t bytes){
 CUdeviceptr base=0;size_t size=0;
 return src&&cuMemGetAddressRange(&base,&size,src)==CUDA_SUCCESS&&src>=base&&src-base<=size&&bytes<=size-(src-base)&&cuMemcpyDtoH(dst,src,bytes)==CUDA_SUCCESS;
}
static CUresult cf_launch(Launch original,CUfunction f,unsigned gx,unsigned bx,CUstream s,void** p){return original(f,gx,1,1,bx,1,1,0,s,p,nullptr);}
static bool cf_allocate(){
 auto allocate=[](CUdeviceptr& p,size_t bytes){return cuMemAlloc(&p,bytes)==CUDA_SUCCESS;};
 return allocate(cf_normal_table,400)&&allocate(cf_friction_table,400)&&allocate(cf_positions,size_t(cf_n)*16)&&allocate(cf_degree,size_t(cf_n)*16);
}
static bool cf_bind(void** params,unsigned gy){
 if(!params||!params[0]||!params[1]||gy!=1)return false;
 CUcontext ctx=nullptr;if(cuCtxGetCurrent(&ctx)!=CUDA_SUCCESS||!ctx)return false;
 CUdeviceptr table=*static_cast<CUdeviceptr*>(params[0]),ids=*static_cast<CUdeviceptr*>(params[1]);
 if(cf_context)return ctx==cf_context&&table==cf_table&&ids==cf_ids;
 if(cuCtxSynchronize()!=CUDA_SUCCESS)return false;
 unsigned id=~0u;unsigned char desc[400];
 if(!cf_read(&id,ids,4)||id!=0||!cf_read(desc,table,400))return false;
 cf_n=cf_get<unsigned>(desc,316);if(cf_n<3||cf_n>100000)return false;
 cf_p=cf_get<CUdeviceptr>(desc,16);cf_v=cf_get<CUdeviceptr>(desc,8);cf_delta=cf_get<CUdeviceptr>(desc,248);
 if(!cf_p||!cf_v||!cf_delta)return false;
 cf_context=ctx;cf_table=table;cf_ids=ids;
 if(cuModuleLoad(&cf_module,cf_ptx.c_str())!=CUDA_SUCCESS)return false;
 size_t count_bytes=0;
 if(cuModuleGetGlobal(&cf_counts,&count_bytes,cf_module,"cloth_split_counts")!=CUDA_SUCCESS||count_bytes!=80||cuMemsetD8(cf_counts,0,80)!=CUDA_SUCCESS)return false;
 const char* names[]={"cloth_split_prepare","cloth_split_bind_frozen","cloth_split_restore_degree","cloth_applyExternalDeltasLaunch","cloth_split_vt_normal","cloth_split_ee_normal","cloth_split_vt_friction","cloth_split_ee_friction","cloth_split_vt_query","cloth_split_ee_query"};
 CUfunction* funcs[]={&cf_prepare,&cf_bind_frozen,&cf_restore,&cf_publish,&cf_normal[0],&cf_normal[1],&cf_friction[0],&cf_friction[1],&cf_query[0],&cf_query[1]};
 for(unsigned i=0;i<10;++i)if(cuModuleGetFunction(funcs[i],cf_module,names[i])!=CUDA_SUCCESS)return false;
 return cf_allocate();
}
static CUresult cf_dispatch(Launch original,const char* name,CUfunction f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned shared,CUstream stream,void** params,void** extra){
 if(!cf_enabled)return original(f,gx,gy,gz,bx,by,bz,shared,stream,params,extra);
 if(cf_errors)return cf_fail();
 if(!std::strcmp(name,"cloth_preIntegrateLaunch")){
  if(cf_pending||!cf_bind(params,gy))return cf_fail();++cf_steps;
 }
 const int kind=!std::strcmp(name,"cloth_solveClothClothDeltaVTLaunch")?0:!std::strcmp(name,"cloth_solveClothClothDeltaEELaunch")?1:-1;
 if(cf_pending&&std::strncmp(name,"cloth_",6)==0){
  if(std::strcmp(name,"cloth_applyExternalDeltasLaunch")||!params||!params[0]||!params[1]||!params[2]||stream!=cf_stream||
     *static_cast<CUdeviceptr*>(params[0])!=cf_table||*static_cast<CUdeviceptr*>(params[1])!=cf_ids||*static_cast<float*>(params[2])!=cf_h)return cf_fail();
  cf_pending=false;++cf_publications;
 }
 if(kind<0)return original(f,gx,gy,gz,bx,by,bz,shared,stream,params,extra);
 CUcontext ctx=nullptr;
 if(!cf_context||cuCtxGetCurrent(&ctx)!=CUDA_SUCCESS||ctx!=cf_context||cf_pending||extra||!params||!params[0]||!params[1]||!params[2]||!params[3]||
    *static_cast<CUdeviceptr*>(params[0])!=cf_table||gy!=1||gz!=1||by!=1||bz!=1||bx!=256||shared!=0)return cf_fail();
 cf_h=*static_cast<float*>(params[3]);if(!(cf_h>0)||!std::isfinite(cf_h))return cf_fail();
 cf_stream=stream;cf_pending=true;++cf_solves[kind];
 if(cf_mode==0)return original(f,gx,gy,gz,bx,by,bz,shared,stream,params,extra);
 void* prepare[]={&cf_table,&cf_normal_table,&cf_friction_table,&cf_positions,&cf_degree,&cf_n,&cf_ids,&cf_p,&cf_v,&cf_delta};
 if(cf_launch(original,cf_prepare,(cf_n+255)/256,256,stream,prepare)!=CUDA_SUCCESS)return cf_fail();
 void* frozen[]={&cf_friction_table,&cf_positions};
 if(cf_launch(original,cf_bind_frozen,1,1,stream,frozen)!=CUDA_SUCCESS)return cf_fail();
 CUdeviceptr rows=*static_cast<CUdeviceptr*>(params[1]),count=*static_cast<CUdeviceptr*>(params[2]);
 void* normal[]={&cf_normal_table,&rows,&count,&cf_h};
 if(cf_launch(original,cf_query[kind],gx,bx,stream,normal)!=CUDA_SUCCESS)return cf_fail();
 if(cf_launch(original,cf_normal[kind],gx,bx,stream,normal)!=CUDA_SUCCESS)return cf_fail();
 void* publish[]={&cf_table,&cf_ids,&cf_h};
 if(cf_launch(original,cf_publish,(cf_n+63)/64,64,stream,publish)!=CUDA_SUCCESS)return cf_fail();
 void* restore[]={&cf_delta,&cf_degree,&cf_n};
 if(cf_launch(original,cf_restore,(cf_n+255)/256,256,stream,restore)!=CUDA_SUCCESS)return cf_fail();
 void* friction[]={&cf_friction_table,&rows,&count,&cf_h};
 if(cf_launch(original,cf_friction[kind],gx,bx,stream,friction)!=CUDA_SUCCESS)return cf_fail();
 return CUDA_SUCCESS;
}
static bool cf_release(){
 if(!cf_context)return true;
 CUcontext current=nullptr;if(cuCtxGetCurrent(&current)!=CUDA_SUCCESS||current!=cf_context)return false;
 if(cuCtxSynchronize()!=CUDA_SUCCESS)return false;
 for(CUdeviceptr* p:{&cf_normal_table,&cf_friction_table,&cf_positions,&cf_degree})if(*p){if(cuMemFree(*p)!=CUDA_SUCCESS)return false;*p=0;}
 if(cf_module&&cuModuleUnload(cf_module)!=CUDA_SUCCESS)return false;cf_module=nullptr;cf_context=nullptr;cf_counts=0;return true;
}
