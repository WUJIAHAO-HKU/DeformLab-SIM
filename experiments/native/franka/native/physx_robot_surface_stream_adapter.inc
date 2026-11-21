// Explicit single-volume / finite horizontal table launch adapter.
// The resident C ABI was independently verified before this native binding.
struct SurfaceBoundaryRow {unsigned node,tet,corner,source;};
extern "C" void* robot_surface_runtime_create(const char*,unsigned,unsigned,const SurfaceBoundaryRow*,unsigned,double,double,double,double,double);
extern "C" int robot_surface_runtime_begin(void*,unsigned);
extern "C" int robot_surface_runtime_native_bind(void*,const CUdeviceptr*,unsigned);
extern "C" int robot_surface_runtime_prepare(void*,const CUdeviceptr*,unsigned,unsigned,CUstream);
extern "C" int robot_surface_runtime_launch_outputs(void*,CUdeviceptr*,unsigned*);
extern "C" int robot_surface_runtime_after_query(void*);
extern "C" int robot_surface_runtime_bind_reaction(void*,CUdeviceptr);
extern "C" int robot_surface_runtime_publish(void*);
extern "C" int robot_surface_runtime_stats(void*,unsigned long long*);
extern "C" int robot_surface_runtime_stop(void*);
namespace {
std::mutex rs_lock;
void* rs_runtime=nullptr;CUdeviceptr rs_ids=0;
unsigned rs_frame=0,rs_round=0,rs_tets=0,rs_expected=0;int rs_error=0;
unsigned long long rs_queries=0,rs_solves=0,rs_frame_queries=0,rs_frame_solves=0,rs_zero_queries=0,rs_final[9]={};
CUdeviceptr rs_input[9]={};unsigned rs_cached_count=0;
CUdeviceptr rs_query_args[11]={},rs_outputs[6]={};float rs_dt=0;unsigned rs_plan[4]={};
int rs_fail(int code,const char* stage){
 if(!rs_error){rs_error=code;std::fprintf(stderr,"Robot surface adapter rejected: stage=%s code=%d frame=%u round=%u queries=%llu solves=%llu\n",stage,code,rs_frame,rs_round,rs_queries,rs_solves);}
 ++errors;return rs_error;
}
bool rs_extent(CUdeviceptr ptr,size_t bytes){
 CUcontext c=nullptr;CUdeviceptr base=0;size_t size=0;
 return ptr&&cuPointerGetAttribute(&c,CU_POINTER_ATTRIBUTE_CONTEXT,ptr)==CUDA_SUCCESS&&c==context&&
  cuMemGetAddressRange(&base,&size,ptr)==CUDA_SUCCESS&&ptr>=base&&bytes<=size&&ptr-base<=size-bytes;
}
CUresult rs_dispatch(Launch original,CUfunction f,bool query,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned shared,CUstream stream,void** params,void** extra){
 std::lock_guard<std::mutex> guard(rs_lock);
 auto fail=[&](int code,const char* stage){rs_fail(code,stage);return CUDA_ERROR_NOT_SUPPORTED;};
 if(rs_error||!rs_frame||!params||extra||!stream||gy!=1||gz!=1||by!=1||bz!=1||!bx||bx>1024||!gx)return fail(401,"launch-contract");
 if(query){
  if(rs_round>=rs_expected||rs_queries!=rs_solves)return fail(402,"query-order");
  if(rs_round){
   for(unsigned i=0;i<11;++i)if(i!=8&&rs_query_args[i]!=*static_cast<CUdeviceptr*>(params[i]))return fail(421,"within-frame-binding");
   if(rs_dt!=*static_cast<float*>(params[8]))return fail(422,"within-frame-dt");
  }
  if(!rs_round){
  for(unsigned i=0;i<11;++i)if(i!=8)rs_query_args[i]=*static_cast<CUdeviceptr*>(params[i]);
  rs_dt=*static_cast<float*>(params[8]);
  if(!(rs_dt>0)||!std::isfinite(rs_dt)||cuStreamSynchronize(stream)!=CUDA_SUCCESS)return fail(403,"query-sync");
  unsigned char soft[432],core[304],art[252];CUdeviceptr mapping=0;
  if(!rs_extent(rs_query_args[0],432)||!rs_extent(rs_query_args[3],4)||!rs_extent(rs_query_args[4],280)||!rs_extent(rs_query_args[5],304)||!rs_extent(rs_query_args[6],252)||
     cuMemcpyDtoH(soft,rs_query_args[0],432)!=CUDA_SUCCESS||cuMemcpyDtoH(core,rs_query_args[5],304)!=CUDA_SUCCESS||cuMemcpyDtoH(art,rs_query_args[6],252)!=CUDA_SUCCESS||
     cuMemcpyDtoH(&mapping,rs_query_args[4]+272,8)!=CUDA_SUCCESS)return fail(404,"native-bindings");
  if(field<unsigned>(soft,372)!=nodes||field<unsigned>(soft,376)!=rs_tets||field<CUdeviceptr>(soft,96)!=positions||field<CUdeviceptr>(soft,88)!=velocities)return fail(405,"single-volume-layout");
  unsigned nb=field<unsigned>(core,268),art_count=field<unsigned>(art,44),links=field<unsigned>(art,248);
  if(!nb||nb>64||art_count>64||links>128)return fail(408,"rigid-bounds");
  CUdeviceptr map_base=0;size_t map_bytes=0;
  if(!mapping||cuMemGetAddressRange(&map_base,&map_bytes,mapping)!=CUDA_SUCCESS||mapping<map_base||mapping-map_base>map_bytes)return fail(409,"solver-map");
  size_t available=(map_bytes-(mapping-map_base))/4;
  unsigned map_capacity=unsigned(std::min(size_t(65536),available));
  // IDs are generated from native contact identities on the GPU in round zero.
  CUdeviceptr input[]={rs_query_args[1],rs_ids,rs_query_args[2],rs_query_args[9],rs_query_args[3],field<CUdeviceptr>(soft,80),positions,field<CUdeviceptr>(soft,128),0};
  std::memcpy(rs_input,input,sizeof(input));rs_cached_count=~0u;
  CUdeviceptr native[]={rs_query_args[0],rs_query_args[4],rs_query_args[5],rs_query_args[6],velocities,mapping};
  int bound=robot_surface_runtime_native_bind(rs_runtime,native,map_capacity);if(bound)return fail(bound,"native-gpu-binding");
  }
  int status=robot_surface_runtime_prepare(rs_runtime,rs_input,rs_frame,rs_round,stream);
  if(status)return fail(status,"prepare");
  if((status=robot_surface_runtime_launch_outputs(rs_runtime,rs_outputs,rs_plan)))return fail(status,"outputs");
  CUdeviceptr values[11];void* args[11];
  std::memcpy(values,rs_query_args,sizeof(values));values[1]=rs_outputs[0];values[2]=rs_outputs[2];values[3]=rs_outputs[5];values[9]=rs_outputs[3];
  for(unsigned i=0;i<11;++i)args[i]=i==8?static_cast<void*>(&rs_dt):static_cast<void*>(&values[i]);
  CUresult c=original(f,std::max(1u,(rs_plan[2]+bx-1)/bx),1,1,bx,1,1,shared,stream,args,nullptr);
  if(c!=CUDA_SUCCESS)return fail(13000+int(c),"native-query");
  if((status=robot_surface_runtime_after_query(rs_runtime)))return fail(status,"query-publication");
  ++rs_queries;if(!rs_cached_count)++rs_zero_queries;return CUDA_SUCCESS;
 }
 if(rs_queries!=rs_solves+1)return fail(413,"solve-order");
 for(unsigned i=0;i<8;++i)if(*static_cast<CUdeviceptr*>(params[i])!=rs_query_args[i])return fail(414,"solve-binding");
 if(*static_cast<CUdeviceptr*>(params[9])!=rs_query_args[9]||*static_cast<CUdeviceptr*>(params[10])!=rs_query_args[10]||*static_cast<float*>(params[11])!=rs_dt)return fail(415,"solve-dt-history");
 int status=robot_surface_runtime_bind_reaction(rs_runtime,*static_cast<CUdeviceptr*>(params[8]));if(status)return fail(status,"reaction-bind");
 CUdeviceptr values[14];void* args[14];float dt=*static_cast<float*>(params[11]);
 for(unsigned i=0;i<14;++i){if(i!=11)values[i]=*static_cast<CUdeviceptr*>(params[i]);args[i]=i==11?static_cast<void*>(&dt):static_cast<void*>(&values[i]);}
 values[1]=rs_outputs[0];values[2]=rs_outputs[2];values[3]=rs_outputs[5];values[8]=rs_outputs[4];values[9]=rs_outputs[3];
 CUresult c=original(f,std::max(1u,(rs_plan[2]+bx-1)/bx),1,1,bx,1,1,shared,stream,args,nullptr);
 if(c!=CUDA_SUCCESS)return fail(14000+int(c),"native-solve");
 if((status=robot_surface_runtime_publish(rs_runtime)))return fail(status,"solve-publication");
 ++rs_solves;++rs_round;return CUDA_SUCCESS;
}
int rs_stop(){
 std::lock_guard<std::mutex> guard(rs_lock);
 if(!rs_runtime)return rs_error;
 if(rs_frame&&(rs_queries-rs_frame_queries!=rs_expected||rs_solves-rs_frame_solves!=rs_expected))rs_fail(416,"final-frame-coverage");
 robot_surface_runtime_stats(rs_runtime,rs_final);
 int code=robot_surface_runtime_stop(rs_runtime);rs_runtime=nullptr;if(code&&!rs_error)rs_fail(code,"stop");
 if(rs_ids){CUresult c=cuCtxPushCurrent(context);if(c==CUDA_SUCCESS){cuMemFree(rs_ids);rs_ids=0;CUcontext previous;cuCtxPopCurrent(&previous);}else rs_fail(417,"ids-stop");}
 return rs_error;
}
}
extern "C" int observer_robot_surface_start(const char* ptx,unsigned ntets,const SurfaceBoundaryRow* boundary,unsigned boundary_count,double cx,double cy,double hx,double hy,double plane,unsigned position_iterations){
 std::lock_guard<std::mutex> guard(rs_lock);
 if(!enabled||rs_runtime||rs_ids||!context||(mode!=2&&mode!=3)||cp_policy||pc_policy||rr_policy||friction_policy||normal_policy||!position_iterations||position_iterations>255)return -1;
 if(cuCtxPushCurrent(context)!=CUDA_SUCCESS)return -2;
 rs_runtime=robot_surface_runtime_create(ptx,unsigned(nodes),ntets,boundary,boundary_count,cx,cy,hx,hy,plane);
 int code=0;if(!rs_runtime)code=-3;else if(cuMemAlloc(&rs_ids,16384*4)!=CUDA_SUCCESS){robot_surface_runtime_stop(rs_runtime);rs_runtime=nullptr;code=-4;}
 CUcontext previous;cuCtxPopCurrent(&previous);if(code)return code;
 rs_frame=rs_round=0;rs_tets=ntets;rs_expected=position_iterations+1;rs_error=0;rs_queries=rs_solves=rs_frame_queries=rs_frame_solves=rs_zero_queries=0;std::memset(rs_final,0,sizeof(rs_final));return 0;
}
extern "C" int observer_robot_surface_frame(unsigned frame){
 std::lock_guard<std::mutex> guard(rs_lock);
 if(!rs_runtime||rs_error||frame!=rs_frame+1)return rs_fail(418,"frame-order");
 if(rs_frame&&(rs_queries-rs_frame_queries!=rs_expected||rs_solves-rs_frame_solves!=rs_expected))return rs_fail(419,"frame-coverage");
 if(cuCtxPushCurrent(context)!=CUDA_SUCCESS)return rs_fail(420,"frame-context");
 int code=robot_surface_runtime_begin(rs_runtime,frame);CUcontext previous;cuCtxPopCurrent(&previous);
 if(code)return rs_fail(code,"frame-reset");rs_frame=frame;rs_round=0;rs_frame_queries=rs_queries;rs_frame_solves=rs_solves;return 0;
}
extern "C" int observer_robot_surface_stats(unsigned long long* out){
 std::lock_guard<std::mutex> guard(rs_lock);if(!out)return -1;
 if(rs_runtime){int code=robot_surface_runtime_stats(rs_runtime,rs_final);if(code)rs_fail(code,"frame-gpu-validation");}
 unsigned long long values[]={rs_queries,rs_solves,rs_frame,rs_round,unsigned(rs_error),rs_zero_queries,rs_ids?16384*4ull:0ull};
 std::memcpy(out,values,sizeof(values));std::memcpy(out+7,rs_final,sizeof(rs_final));return 0;
}

extern "C" int robot_surface_runtime_component_trace(void*,unsigned long long*,unsigned);
extern "C" int observer_robot_surface_component_trace(unsigned long long* out,unsigned rounds){
 std::lock_guard<std::mutex> guard(rs_lock);
 if(!rs_runtime||!out||rounds!=rs_expected||rs_round!=rs_expected)return -1;
 return robot_surface_runtime_component_trace(rs_runtime,out,rounds);
}
