// Read-only, synchronized snapshots of two native volumetric bodies.
// Included inside the launch profiler's namespace; host launch mutex is held.
struct MsBody {unsigned id=0,n=0,nt=0;CUdeviceptr address=0;};
FILE* ms_index=nullptr;gzFile ms_blob=nullptr;CUcontext ms_context=nullptr;
std::vector<MsBody> ms_bodies;unsigned ms_step=0,ms_first=0,ms_last=0,ms_average=0,ms_errors=0;
unsigned long long ms_seq=0,ms_records=0,ms_offset=0;bool ms_failed=false;
template<class T>T ms_field(const unsigned char* p,size_t offset){T v;std::memcpy(&v,p+offset,sizeof(v));return v;}
bool ms_failure(const char* reason){if(!ms_failed){++ms_errors;++errors;std::fprintf(stderr,"Multisoft observation rejected: %s at step %u\n",reason,ms_step);}ms_failed=true;return false;}
bool ms_copy(void* host,CUdeviceptr device,size_t bytes){
 if(!bytes)return true;if(!device||bytes>64*1024*1024)return ms_failure("invalid pointer/size");
 CUdeviceptr base=0;size_t capacity=0;CUcontext owner=nullptr;
 if(cuPointerGetAttribute(&owner,CU_POINTER_ATTRIBUTE_CONTEXT,device)!=CUDA_SUCCESS||owner!=ms_context||
    cuMemGetAddressRange(&base,&capacity,device)!=CUDA_SUCCESS||device<base||device-base>capacity||bytes>capacity-size_t(device-base))return ms_failure("device allocation range/context");
 if(cuMemcpyDtoH(host,device,bytes)!=CUDA_SUCCESS)return ms_failure("device readback");return true;
}
bool ms_write(const char* key,const void* data,size_t bytes,bool& first){
 if(ms_offset+bytes>4ull*1024*1024*1024)return ms_failure("uncompressed observation capacity");
 if(bytes&&gzwrite(ms_blob,data,unsigned(bytes))!=int(bytes))return ms_failure("compressed write");
 if(std::fprintf(ms_index,"%s\"%s\":[%llu,%zu]",first?"":",",key,ms_offset,bytes)<0)return ms_failure("index write");
 first=false;ms_offset+=bytes;return true;
}
bool ms_read_write(const char* key,CUdeviceptr address,size_t bytes,bool& first){
 std::vector<unsigned char> data(bytes);return ms_copy(data.data(),address,bytes)&&ms_write(key,data.data(),bytes,first);
}
bool ms_discover(void** params,unsigned gy){
 if(!params||!params[0]||!params[1]||gy!=2)return ms_failure("expected two active bodies at initial discovery");
 if(cuCtxGetCurrent(&ms_context)!=CUDA_SUCCESS||!ms_context||cuCtxSynchronize()!=CUDA_SUCCESS)return ms_failure("initial context");
 CUdeviceptr table=*static_cast<CUdeviceptr*>(params[0]),ids=*static_cast<CUdeviceptr*>(params[1]);unsigned index[2];
 if(!ms_copy(index,ids,sizeof(index))||index[0]==index[1])return ms_failure("initial body indices");
 for(unsigned id:index){
  if(id>=4096)return ms_failure("body table index");
  unsigned char meta[432];MsBody b;b.id=id;b.address=table+size_t(id)*432;
  if(!ms_copy(meta,b.address,sizeof(meta)))return false;
  b.n=ms_field<unsigned>(meta,372);b.nt=ms_field<unsigned>(meta,376);
  if(!b.n||b.n>100000||!b.nt||b.nt>1000000)return ms_failure("body dimensions");
  std::vector<unsigned> tets(size_t(b.nt)*4);
  if(!ms_copy(tets.data(),ms_field<CUdeviceptr>(meta,80),tets.size()*4))return false;
  for(unsigned node:tets)if(node>=b.n)return ms_failure("tetrahedron node index");
  std::fprintf(ms_index,"{\"kind\":\"topology\",\"body\":%u,\"nodes\":%u,\"tets\":%u,\"sections\":{",id,b.n,b.nt);bool first=true;
  if(!ms_write("tet_indices",tets.data(),tets.size()*4,first))return false;
  std::fprintf(ms_index,"}}\n");ms_bodies.push_back(b);
 }
 return true;
}
bool ms_selected(const char* name){
 return !std::strcmp(name,"sb_gm_preIntegrateLaunch")||!std::strcmp(name,"sb_gm_cp_averageVertsLaunch")||
 !std::strcmp(name,"sb_gm_stepSoftbodyLaunch")||!std::strcmp(name,"sb_gm_finalizeVelocitiesLaunch")||
 !std::strcmp(name,"sb_gm_applyExternalDeltasLaunch")||!std::strcmp(name,"sb_solveOutputSSDeltaVLaunchTGS")||
 !std::strcmp(name,"sb_queryRigidSoftContactReferenceCountLaunchTGS")||!std::strcmp(name,"sb_solveRigidSoftCollisionLaunchTGS");
}
bool ms_snapshot(const char* name,void** params,bool before,unsigned long long seq){
 if(cuCtxSynchronize()!=CUDA_SUCCESS)return ms_failure("snapshot synchronization");
 for(const auto& b:ms_bodies){
  unsigned char meta[432];if(!ms_copy(meta,b.address,sizeof(meta)))return false;
  if(ms_field<unsigned>(meta,372)!=b.n||ms_field<unsigned>(meta,376)!=b.nt)return ms_failure("body dimensions changed");
  std::fprintf(ms_index,"{\"kind\":\"state\",\"seq\":%llu,\"step\":%u,\"average_ordinal\":%u,\"kernel\":\"%s\",\"phase\":\"%s\",\"body\":%u,\"nodes\":%u,\"tets\":%u,\"sections\":{",seq,ms_step,ms_average,name,before?"before":"after",b.id,b.n,b.nt);bool first=true;
  if(!ms_write("metadata",meta,sizeof(meta),first)||
     !ms_read_write("position",ms_field<CUdeviceptr>(meta,96),size_t(b.n)*16,first)||
     !ms_read_write("velocity",ms_field<CUdeviceptr>(meta,88),size_t(b.n)*16,first)||
     !ms_read_write("delta_position",ms_field<CUdeviceptr>(meta,128),size_t(b.n)*16,first)||
     !ms_read_write("contact_delta",ms_field<CUdeviceptr>(meta,272),size_t(b.n)*16,first))return false;
  std::fprintf(ms_index,"}}\n");++ms_records;
 }
 bool ss=!std::strcmp(name,"sb_solveOutputSSDeltaVLaunchTGS");
 bool rigid=!std::strcmp(name,"sb_solveRigidSoftCollisionLaunchTGS");
 if(ss||rigid){
  if(!params||!params[3])return ms_failure("contact parameters");
  unsigned count=0;if(!ms_copy(&count,*static_cast<CUdeviceptr*>(params[3]),4)||count>16384)return ms_failure("contact row count");
  float dt=*static_cast<float*>(params[ss?4:11]);
  std::fprintf(ms_index,"{\"kind\":\"contact\",\"seq\":%llu,\"step\":%u,\"average_ordinal\":%u,\"kernel\":\"%s\",\"phase\":\"%s\",\"contacts\":%u,\"scalar_parameter_value\":%.17g,\"sections\":{",seq,ms_step,ms_average,name,before?"before":"after",count,double(dt));bool first=true;
  if(!ms_read_write("info",*static_cast<CUdeviceptr*>(params[1]),size_t(count)*16,first)||
     !ms_read_write("constraints",*static_cast<CUdeviceptr*>(params[2]),size_t((count+31)/32)*(ss?1792:3200),first))return false;
  std::fprintf(ms_index,"}}\n");
 }
 return !std::ferror(ms_index)||ms_failure("index stream");
}
bool ms_before(const char* name,void** params,unsigned gy,unsigned long long& seq){
 if(!ms_index||ms_failed)return false;
 if(!std::strcmp(name,"sb_gm_preIntegrateLaunch")){
  ++ms_step;ms_average=0;
  if(ms_bodies.empty()&&!ms_discover(params,gy))return false;
 }
 if(ms_step<ms_first||ms_step>ms_last||!ms_selected(name))return false;
 CUcontext current=nullptr;if(cuCtxGetCurrent(&current)!=CUDA_SUCCESS||current!=ms_context)return ms_failure("snapshot context changed");
 if(!std::strcmp(name,"sb_gm_cp_averageVertsLaunch"))++ms_average;
 seq=ms_seq++;return ms_snapshot(name,params,true,seq);
}
void ms_after(const char* name,void** params,unsigned long long seq){if(!ms_failed)ms_snapshot(name,params,false,seq);}
