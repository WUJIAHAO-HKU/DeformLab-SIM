// Inserted after cf_allocate; all state belongs to one pinned cloth lifecycle.
static bool fh_enabled=false,fh_has_stream=false,fh_has_publication=false;
static CUevent fh_snapshot_ready=nullptr,fh_publication_ready=nullptr;
static unsigned long long fh_order[5]{};
static unsigned fh_capacity=32768,fh_limit=100000,fh_epoch=0,fh_nt=0;
static unsigned long long fh_frames=0,fh_updates=0,fh_last[16]{};
static CUdeviceptr fh_history=0,fh_previous=0,fh_slots=0,fh_counters=0,fh_errors=0,fh_triangles=0;
static CUfunction fh_capture=nullptr,fh_resolve=nullptr,fh_update=nullptr,fh_guard=nullptr;
static CUstream fh_stream=nullptr;
static void fh_reset(){fht_reset();fh_enabled=fh_has_stream=fh_has_publication=false;for(auto& v:fh_order)v=0;fh_epoch=fh_nt=0;fh_frames=fh_updates=0;fh_stream=nullptr;fh_triangles=0;for(auto& v:fh_last)v=0;}
static bool fh_allocate(){
 if(!fh_enabled)return cf_mode==0;
 if(cf_n>65536||!eq_enabled)return false;
 unsigned char desc[400];if(!cf_read(desc,cf_table,400))return false;
 fh_nt=cf_get<unsigned>(desc,320);fh_triangles=cf_get<CUdeviceptr>(desc,72);
 if(!fh_nt||fh_nt>200000||!fh_triangles||!fht_bind())return false;
 const char* names[]={"cloth_history_capture_previous","cloth_history_resolve_indirect","cloth_ee_history_update","cloth_history_guard"};CUfunction* functions[]={&fh_capture,&fh_resolve,&fh_update,&fh_guard};
 for(unsigned i=0;i<4;++i)if(cuModuleGetFunction(functions[i],cf_module,names[i])!=CUDA_SUCCESS)return false;
 for(auto item:{std::pair<CUdeviceptr*,size_t>{&fh_history,size_t(fh_capacity)*24},{&fh_previous,size_t(cf_n)*16},{&fh_slots,size_t(fh_limit)*4},{&fh_counters,64},{&fh_errors,4}}){if(cuMemAlloc(item.first,item.second)!=CUDA_SUCCESS||cuMemsetD8(*item.first,0,item.second)!=CUDA_SUCCESS)return false;}
 if(cuEventCreate(&fh_snapshot_ready,CU_EVENT_DISABLE_TIMING)!=CUDA_SUCCESS)return false;++fh_order[3];
 if(cuEventCreate(&fh_publication_ready,CU_EVENT_DISABLE_TIMING)!=CUDA_SUCCESS)return false;++fh_order[3];
 return true;
}
static bool fh_frame(Launch original,CUstream stream){
 if(!fh_enabled)return cf_mode==0;
 if(fh_has_publication){if(cuStreamWaitEvent(stream,fh_publication_ready,0)!=CUDA_SUCCESS)return false;++fh_order[2];}
 fh_has_stream=true;fh_stream=stream;++fh_frames;
 void* args[]={&cf_table,&fh_previous,&cf_n,&cf_p,&fh_triangles,&fh_nt,&fh_errors,&fh_counters};
 if(cf_launch(original,fh_capture,(cf_n+255)/256,256,stream,args)!=CUDA_SUCCESS||cuEventRecord(fh_snapshot_ready,stream)!=CUDA_SUCCESS)return false;++fh_order[0];return true;
}
static bool fh_wait(CUstream stream){
 if(!fh_enabled)return cf_mode==0;
 if(!fh_has_stream||cuStreamWaitEvent(stream,fh_snapshot_ready,0)!=CUDA_SUCCESS)return false;++fh_order[2];
 if(fh_has_publication){if(cuStreamWaitEvent(stream,fh_publication_ready,0)!=CUDA_SUCCESS)return false;++fh_order[2];}
 return true;
}
static bool fh_publication(CUstream stream){
 if(!fh_enabled)return true;
 if(cuEventRecord(fh_publication_ready,stream)!=CUDA_SUCCESS)return false;fh_has_publication=true;++fh_order[1];return true;
}
static bool fh_solve(Launch original,CUdeviceptr rows,CUdeviceptr count,CUstream stream){
 if(!fh_enabled||!fh_has_stream||fh_epoch==~0u)return false;++fh_epoch;++fh_updates;
 void* resolve[]={&rows,&count,&fh_triangles,&fh_nt,&cf_n,&fh_history,&fh_capacity,&fh_epoch,&fh_slots,&fh_counters};
 if(cf_launch(original,fh_resolve,128,256,stream,resolve)!=CUDA_SUCCESS)return false;
 double release=.0032;void* update[]={&fh_history,&fh_capacity,&fh_previous,&cf_p,&cf_n,&fh_epoch,&release,&fh_errors};
 if(cf_launch(original,fh_update,(fh_capacity+255)/256,256,stream,update)!=CUDA_SUCCESS)return false;
 void* guard[]={&fh_errors,&fh_counters,&cf_p,&fh_previous,&fh_history,&fh_capacity,&fh_epoch,&cf_n};return cf_launch(original,fh_guard,1,1,stream,guard)==CUDA_SUCCESS;
}
static bool fh_report(){
 if(!fh_enabled)return true;
 fh_last[0]=fh_frames;fh_last[1]=fh_updates;unsigned error=0;
 if(!cf_context||cuCtxSynchronize()!=CUDA_SUCCESS||!cf_read(fh_last+2,fh_counters,64)||!cf_read(&error,fh_errors,4)){fh_last[15]=1;return false;}
 fh_last[10]=error;fh_last[11]=cf_n;fh_last[12]=fh_nt;fh_last[13]=fh_capacity;fh_last[14]=24ull*fh_capacity+16ull*cf_n+4ull*fh_limit+68;
 return !error&&!fh_last[6]&&fh_last[0]==cf_steps&&fh_last[1]==cf_solves[1]&&fh_last[7]==fh_updates&&fh_last[8]==fh_frames;
}
static bool fh_release(){
 for(CUevent* e:{&fh_snapshot_ready,&fh_publication_ready})if(*e){if(cuEventDestroy(*e)!=CUDA_SUCCESS)return false;*e=nullptr;++fh_order[4];}
 for(CUdeviceptr* p:{&fh_history,&fh_previous,&fh_slots,&fh_counters,&fh_errors})if(*p){if(cuMemFree(*p)!=CUDA_SUCCESS)return false;*p=0;}
 return true;
}
