// Optional asynchronous stage observer. Never writes solver state.
struct BGHostDesc {CUdeviceptr p=0,t=0,reference=0,previous=0;unsigned n=0,nt=0;};
static_assert(sizeof(BGHostDesc)==40,"Geometry descriptor ABI");
bool bg_enabled=false,bg_ready=false,bg_event_recorded=false;
std::string bg_ptx,bg_prefix;CUmodule bg_module=nullptr;CUfunction bg_function=nullptr;CUevent bg_event=nullptr;
CUdeviceptr bg_table=0,bg_stats=0;BGHostDesc bg_bodies[2];std::vector<double> bg_reference[2];
unsigned long long bg_calls[8]{},bg_errors=0;
bool bg_fail(const char* why){++bg_errors;++errors;std::fprintf(stderr,"Batched geometry rejected: %s\n",why);return false;}
bool bg_wait(const char* name,CUstream stream){
 if(!bg_enabled||!bg_event_recorded||std::strncmp(name,"sb_",3))return true;
 return cuStreamWaitEvent(stream,bg_event,0)==CUDA_SUCCESS||bg_fail("stage ordering event");
}
bool bg_initialize(CUstream stream){
 if(bg_ready)return true;if(!bp_context||!bp_table)return bg_fail("mapping must bind first");
 if(cuModuleLoad(&bg_module,bg_ptx.c_str())!=CUDA_SUCCESS||cuModuleGetFunction(&bg_function,bg_module,"batched_geometry_check")!=CUDA_SUCCESS||cuEventCreate(&bg_event,CU_EVENT_DISABLE_TIMING)!=CUDA_SUCCESS)return bg_fail("observer initialization");
 for(unsigned j=0;j<2;++j){auto& b=bp_body[j];auto& g=bg_bodies[j];unsigned char meta[432];
  if(!bp_read(meta,bp_table+size_t(b.id)*432,432))return bg_fail("topology metadata");
  g.p=b.d.p;g.t=bp_field<CUdeviceptr>(meta,80);g.n=b.d.n;g.nt=b.d.nt;
  std::vector<unsigned> t(size_t(g.nt)*4);std::vector<float> p(size_t(g.n)*4);
  if(!bp_read(t.data(),g.t,t.size()*4)||!bp_read(p.data(),g.p,p.size()*4))return bg_fail("topology/root range");
  auto& ref=bg_reference[j];ref.resize(g.nt);
  for(unsigned k=0;k<g.nt;++k){unsigned ids[4];for(unsigned a=0;a<4;++a){ids[a]=t[4*k+a];if(ids[a]>=g.n)return bg_fail("tet index outside body");}
   double e[3][3];for(unsigned a=0;a<3;++a)for(unsigned c=0;c<3;++c)e[a][c]=double(p[4*ids[a+1]+c])-p[4*ids[0]+c];
   double value=e[0][0]*(e[1][1]*e[2][2]-e[1][2]*e[2][1])+e[0][1]*(e[1][2]*e[2][0]-e[1][0]*e[2][2])+e[0][2]*(e[1][0]*e[2][1]-e[1][1]*e[2][0]);
   if(!std::isfinite(value)||value<=0)return bg_fail("positive initial reference required");ref[k]=value;
  }
  if(cuMemAlloc(&g.reference,size_t(g.nt)*8)!=CUDA_SUCCESS||cuMemAlloc(&g.previous,size_t(g.n)*16)!=CUDA_SUCCESS||cuMemcpyHtoD(g.reference,ref.data(),ref.size()*8)!=CUDA_SUCCESS||cuMemcpyDtoDAsync(g.previous,g.p,size_t(g.n)*16,stream)!=CUDA_SUCCESS)return bg_fail("body observer allocation");
 }
 if(cuMemAlloc(&bg_table,sizeof(bg_bodies))!=CUDA_SUCCESS||cuMemcpyHtoD(bg_table,bg_bodies,sizeof(bg_bodies))!=CUDA_SUCCESS||cuMemAlloc(&bg_stats,16*192)!=CUDA_SUCCESS||cuMemsetD8(bg_stats,0,16*192)!=CUDA_SUCCESS)return bg_fail("observer descriptor/statistics");
 bg_ready=true;return true;
}
bool bg_pre(const char* name,CUstream stream){
 if(!bg_enabled||std::strcmp(name,"sb_gm_preIntegrateLaunch"))return true;
 return bg_initialize(stream);
}
unsigned bg_stage(const char* name,bool mapped){
 if(mapped)return 3;
 if(!std::strcmp(name,"sb_gm_preIntegrateLaunch"))return 0;
 if(!std::strcmp(name,"sb_gm_cp_solveTetrahedronsPartitionLaunch"))return 1;
 if(!std::strcmp(name,"sb_gm_cp_averageVertsLaunch"))return 2;
 if(!std::strcmp(name,"sb_gm_applyExternalDeltasLaunch"))return 4;
 if(!std::strcmp(name,"sb_gm_stepSoftbodyLaunch"))return 5;
 if(!std::strcmp(name,"sb_gm_finalizeVelocitiesLaunch"))return 6;
 return 7;
}
bool bg_observe(const char* name,CUstream stream,Launch original,bool mapped){
 if(!bg_enabled||!bg_ready||std::strncmp(name,"sb_",3))return true;
 unsigned count=2,stages=8,stage=bg_stage(name,mapped);void* args[]={&bg_table,&count,&stage,&stages,&bp_frames,&bp_cycles,&bg_stats};
 if(original(bg_function,1,2,1,128,1,1,0,stream,args,nullptr)!=CUDA_SUCCESS||cuEventRecord(bg_event,stream)!=CUDA_SUCCESS)return bg_fail("geometry observation launch/event");
 bg_event_recorded=true;++bg_calls[stage];return true;
}
bool bg_write(const std::string& path,const void* data,size_t bytes){FILE* f=std::fopen(path.c_str(),"wx");if(!f)return false;bool ok=std::fwrite(data,1,bytes,f)==bytes;if(std::fclose(f))ok=false;return ok;}
