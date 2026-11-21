// Experimental two-volume SS binding. Exactly the captured native ABI is accepted.
// All helper launches use the original driver entry and the native solve stream.
std::atomic<bool> ss_enabled{false};int ss_mode=0;
std::string ss_ptx;CUcontext ss_context=nullptr;CUmodule ss_module=nullptr;CUfunction ss_function=nullptr;
CUdeviceptr ss_previous=0,ss_stats=0,ss_table=0,ss_history=0;
struct SSBody {unsigned id=0,n=0;CUdeviceptr pending=0;};SSBody ss_bodies[2];
unsigned ss_count=0;unsigned long long ss_steps=0,ss_calls=0,ss_applies=0,ss_errors=0;
bool ss_pending=false,ss_apply_pending=false;
bool ss_fail(const char* reason){++ss_errors;++errors;std::fprintf(stderr,"SS common binding rejected: %s\n",reason);return false;}
bool ss_range(CUdeviceptr p,size_t bytes){
 if(!bytes)return true;CUdeviceptr base=0;size_t capacity=0;CUcontext owner=nullptr;
 return p&&cuPointerGetAttribute(&owner,CU_POINTER_ATTRIBUTE_CONTEXT,p)==CUDA_SUCCESS&&owner==ss_context&&cuMemGetAddressRange(&base,&capacity,p)==CUDA_SUCCESS&&p>=base&&p-base<=capacity&&bytes<=capacity-size_t(p-base);
}
template<class T>T ss_field(const unsigned char* p,size_t offset){T result;std::memcpy(&result,p+offset,sizeof(result));return result;}
bool ss_read(void* host,CUdeviceptr ptr,size_t bytes){return ss_range(ptr,bytes)&&cuMemcpyDtoH(host,ptr,bytes)==CUDA_SUCCESS;}
bool ss_before(const char* name,void** params,unsigned gy){
 if(!ss_enabled)return true;if(ss_errors)return false;
 CUcontext current=nullptr;if(cuCtxGetCurrent(&current)!=CUDA_SUCCESS||!current)return ss_fail("missing CUDA context");
 if(ss_context&&current!=ss_context)return ss_fail("context changed");
 if(ss_apply_pending&&std::strcmp(name,"sb_gm_applyExternalDeltasLaunch"))return ss_fail("SS publication order changed");
 if(!std::strcmp(name,"sb_gm_preIntegrateLaunch")){
  if(!params||!params[0]||!params[1]||gy!=2)return ss_fail("expected exactly two active volume bodies");
  if(!ss_context){
   ss_context=current;
   if(cuModuleLoad(&ss_module,ss_ptx.c_str())!=CUDA_SUCCESS||cuModuleGetFunction(&ss_function,ss_module,"multisoft_common_history")!=CUDA_SUCCESS||cuMemAlloc(&ss_previous,16384*8)!=CUDA_SUCCESS||cuMemAlloc(&ss_stats,6*8)!=CUDA_SUCCESS||cuMemsetD8(ss_stats,0,6*8)!=CUDA_SUCCESS)return ss_fail("helper initialization");
  }
  if(cuCtxSynchronize()!=CUDA_SUCCESS)return ss_fail("initial state synchronization");
  CUdeviceptr table=*static_cast<CUdeviceptr*>(params[0]);unsigned ids[2];
  if(ss_table&&ss_table!=table)return ss_fail("body table changed");
  if(!ss_read(ids,*static_cast<CUdeviceptr*>(params[1]),8)||ids[0]==ids[1]||ids[0]>=4096||ids[1]>=4096)return ss_fail("active body indices");
  for(unsigned j=0;j<2;++j){
   unsigned char metadata[432];if(!ss_read(metadata,table+size_t(ids[j])*432,432))return ss_fail("body metadata range");
   SSBody b;b.id=ids[j];b.n=ss_field<unsigned>(metadata,372);b.pending=ss_field<CUdeviceptr>(metadata,272);
   if(!b.n||b.n>100000||!ss_range(b.pending,size_t(b.n)*16))return ss_fail("pending buffer range");
   if(ss_steps&&(b.id!=ss_bodies[j].id||b.n!=ss_bodies[j].n||b.pending!=ss_bodies[j].pending))return ss_fail("body identity changed inside scene");
   ss_bodies[j]=b;
  }
  auto a=ss_bodies[0].pending,b=ss_bodies[1].pending;
  if(a<b+size_t(ss_bodies[1].n)*16&&b<a+size_t(ss_bodies[0].n)*16)return ss_fail("overlapping pending buffers");
  ss_table=table;++ss_steps;
 }
 if(!std::strcmp(name,"sb_solveOutputSSDeltaVLaunchTGS")){
  if(!ss_steps||ss_pending||ss_apply_pending||!params||!params[0]||!params[3]||!params[6]||*static_cast<CUdeviceptr*>(params[0])!=ss_table)return ss_fail("unbound SS solve");
  if(cuCtxSynchronize()!=CUDA_SUCCESS||!ss_read(&ss_count,*static_cast<CUdeviceptr*>(params[3]),4)||ss_count>16384)return ss_fail("SS contact count");
  ss_history=*static_cast<CUdeviceptr*>(params[6]);
  if(!ss_range(ss_history,size_t(ss_count)*8))return ss_fail("history allocation range");
  if(ss_count&&cuMemcpyDtoD(ss_previous,ss_history,size_t(ss_count)*8)!=CUDA_SUCCESS)return ss_fail("history checkpoint");
  ss_pending=true;
 }
 if(ss_apply_pending){
  if(!params||!params[0]||*static_cast<CUdeviceptr*>(params[0])!=ss_table||gy!=2)return ss_fail("publication body table");
 }
 return true;
}
bool ss_after(const char* name,CUstream stream,Launch original){
 if(!ss_enabled)return true;
 if(!std::strcmp(name,"sb_solveOutputSSDeltaVLaunchTGS")){
  if(!ss_pending)return ss_fail("missing SS checkpoint");
  void* args[]={&ss_bodies[0].pending,&ss_bodies[0].n,&ss_bodies[1].pending,&ss_bodies[1].n,&ss_mode,&ss_stats,&ss_previous,&ss_history,&ss_count};
  if(original(ss_function,1,1,1,128,1,1,0,stream,args,nullptr)!=CUDA_SUCCESS||cuCtxSynchronize()!=CUDA_SUCCESS)return ss_fail("common update launch");
  unsigned long long stats[6];if(cuMemcpyDtoH(stats,ss_stats,sizeof(stats))!=CUDA_SUCCESS||stats[1])return ss_fail("invalid common-update input");
  ss_pending=false;ss_apply_pending=true;++ss_calls;
 }else if(ss_apply_pending&&!std::strcmp(name,"sb_gm_applyExternalDeltasLaunch")){
  ss_apply_pending=false;++ss_applies;
 }
 return true;
}
