// Read-only, opt-in observation of the pinned TGS contact path. Context-wide
// synchronization is intentionally intrusive; compare with unobserved runs.
namespace {
FILE* ct_index=nullptr;FILE* ct_blob=nullptr;std::mutex ct_lock;
std::atomic<int> ct_step{-1};int ct_round=-1;unsigned long long ct_seq=0,ct_records=0;
bool ct_valid=false;CUdeviceptr ct_args[14]={};float ct_dt=0;
bool ct_detail=false,ct_coupon=false;
bool ct_selected(const char* name){
 return (ct_coupon && (!std::strcmp(name,"artiPushImpulse") || !std::strcmp(name,"artiOutputVelocity") || !std::strcmp(name,"stepArticulation1TTGS"))) ||
 !std::strcmp(name,"sb_solveRigidSoftCollisionLaunchTGS") ||
 !std::strcmp(name,"accumulateDeltaVRigidSecondLaunch") ||
 !std::strcmp(name,"sb_gm_applyExternalDeltasLaunch") ||
 !std::strcmp(name,"computeAverageSolverBodyVelocityTGS") ||
 !std::strcmp(name,"propagateAverageSolverBodyVelocityTGS") ||
 (ct_detail && (!std::strcmp(name,"sb_gm_cp_averageVertsLaunch")||!std::strcmp(name,"sb_gm_stepSoftbodyLaunch")||!std::strcmp(name,"sb_gm_finalizeVelocitiesLaunch")));
}
bool ct_snapshot(const char* name,void** params,bool before,unsigned long long seq){
 if(!ct_index || ct_step<0 || !enabled.load())return false;
 CUcontext current=nullptr;if(cuCtxGetCurrent(&current)!=CUDA_SUCCESS || current!=context)return false;
 auto fail=[&](){++errors;std::fprintf(stderr,"Contact observation rejected: %s\n",name);return false;};
 if(cuCtxSynchronize()!=CUDA_SUCCESS)return fail();
 bool solve=!std::strcmp(name,"sb_solveRigidSoftCollisionLaunchTGS");
 if(solve && before){
  if(!params || !sb || (mode<9 && !(ct_coupon && mode==6)))return fail();
  for(int i=0;i<14;++i)if(i!=11)ct_args[i]=*static_cast<CUdeviceptr*>(params[i]);
  ct_dt=*static_cast<float*>(params[11]);ct_valid=true;++ct_round;
 }
 if(!ct_valid)return false;
 unsigned c=0;unsigned char core[304],shared[224],soft[432],arti[252]={};
 if(cuMemcpyDtoH(&c,ct_args[3],4)!=CUDA_SUCCESS || c>16384 ||
 cuMemcpyDtoH(core,ct_args[5],304)!=CUDA_SUCCESS || cuMemcpyDtoH(shared,ct_args[7],224)!=CUDA_SUCCESS ||
 cuMemcpyDtoH(soft,sb,432)!=CUDA_SUCCESS)return fail();
 unsigned nb=field<unsigned>(core,268),offset=field<unsigned>(core,276),nt=field<unsigned>(soft,376);
 if(!nb || nb>64 || offset>4096 || !nt || nt>1000000 || !(ct_dt>0) || nodes!=field<unsigned>(soft,372))return fail();
 unsigned art_count=0,max_links=0,art_capacity=0;
 if(ct_coupon){
  if(cuMemcpyDtoH(arti,ct_args[6],sizeof(arti))!=CUDA_SUCCESS)return fail();
  art_count=field<unsigned>(arti,44);max_links=field<unsigned>(arti,248);
  if(art_count>64 || max_links>128 || size_t(art_count)*max_links>4096)return fail();
  art_capacity=art_count*max_links;
 }
 std::vector<unsigned char> info(size_t(c)*16);std::vector<unsigned> ids(c);CUdeviceptr index_ptr=0;
 if(c && cuMemcpyDtoH(info.data(),ct_args[1],info.size())!=CUDA_SUCCESS)return fail();
 if(cuMemcpyDtoH(&index_ptr,ct_args[4]+272,8)!=CUDA_SUCCESS)return fail();
 for(unsigned i=0;i<c;++i){auto row=info.data()+16*i;unsigned rigid=field<unsigned>(row,0),flags=field<unsigned>(row,4),packed=field<unsigned>(row,8);
  if((packed&0xfffffu)>=nt || ct_args[0]+size_t(packed>>20)*432!=sb)return fail();
  if(rigid==~0u){if(!ct_coupon)return fail();ids[i]=~0u;continue;}
  if(cuMemcpyDtoH(&ids[i],index_ptr+size_t(rigid)*4,4)!=CUDA_SUCCESS)return fail();
  if(flags&1u){
   if(!ct_coupon || ids[i]>=art_count || (flags>>1)>=max_links)return fail();
   ids[i]=nb+ids[i]*max_links+(flags>>1);
  }else if(ids[i]>=nb)return fail();
 }
 std::fprintf(ct_index,"{\"seq\":%llu,\"step\":%d,\"round\":%d,\"kernel\":\"%s\",\"phase\":\"%s\",\"dt\":%.17g,\"nodes\":%zu,\"tets\":%u,\"contacts\":%u,\"bodies\":%u,\"pool_offset\":%u,\"sections\":{",seq,ct_step.load(),ct_round,name,before?"before":"after",double(ct_dt),nodes,nt,c,nb,offset);
 bool first=true,ok=true;
 auto host=[&](const char* key,const void* data,size_t size){
  if(!ok)return;long start=std::ftell(ct_blob);if(start<0 || (size && std::fwrite(data,1,size,ct_blob)!=size)){ok=false;return;}
  if(!first)std::fprintf(ct_index,",");first=false;std::fprintf(ct_index,"\"%s\":[%ld,%zu]",key,start,size);
 };
 auto device=[&](const char* key,CUdeviceptr ptr,size_t size){
  std::vector<unsigned char> data(size);if(size && cuMemcpyDtoH(data.data(),ptr,size)!=CUDA_SUCCESS){ok=false;return;}host(key,data.data(),size);
 };
 host("info",info.data(),info.size());host("solver_ids",ids.data(),ids.size()*4);
 host("solver_core",core,sizeof(core));host("solver_shared",shared,sizeof(shared));host("soft_meta",soft,sizeof(soft));
 if(ct_coupon){
  host("articulation_core",arti,sizeof(arti));
  device("articulation_velocity",field<CUdeviceptr>(core,40),size_t(art_capacity)*4*16);
 }
 device("constraints",ct_args[2],size_t((c+31)/32)*3200);
 device("lambda",ct_args[9],size_t(c)*4);device("rigid_delta",ct_args[8],size_t(c)*32);
 device("rigid_counts",ct_args[10],size_t(nb+art_capacity)*4);
 device("body_data",field<CUdeviceptr>(core,16),size_t(nb)*96);
 device("body_txi",field<CUdeviceptr>(core,24),size_t(nb)*64);
 device("body_pool",field<CUdeviceptr>(shared,160),size_t(offset+3*nb)*16);
 device("tet_indices",field<CUdeviceptr>(soft,80),size_t(nt)*16);
 device("position",positions,nodes*16);device("velocity",velocities,nodes*16);
 device("delta_position",field<CUdeviceptr>(soft,128),nodes*16);device("contact_delta",field<CUdeviceptr>(soft,272),nodes*16);
 if(mode>=9)device("residual",residual,nodes*72);
 if(rr_policy)device("rigid_reduction_state",rr_state,64*sizeof(RigidReductionState));
 std::fprintf(ct_index,"}}\n");if(!ok || std::ferror(ct_index))return fail();++ct_records;return true;
}
}
extern "C" int observer_contact_start(const char* path){
 if(ct_index || !enabled || mode<9)return -1;
 ct_index=std::fopen((std::string(path)+".jsonl").c_str(),"wx");
 ct_blob=std::fopen((std::string(path)+".bin").c_str(),"wx");return ct_index && ct_blob?0:-2;
}
extern "C" int observer_contact_coupon_start(const char* path){
 if(ct_index || !enabled || mode!=6)return -1;
 ct_coupon=true;ct_detail=true;ct_records=ct_seq=0;
 ct_index=std::fopen((std::string(path)+".jsonl").c_str(),"wx");
 ct_blob=std::fopen((std::string(path)+".bin").c_str(),"wx");return ct_index && ct_blob?0:-2;
}
extern "C" void observer_contact_step(int step){ct_step=step;ct_valid=false;ct_round=-1;}
extern "C" void observer_contact_detail(int detail){ct_detail=detail!=0;}
extern "C" unsigned long long observer_contact_stop(){
 ct_step=-1;ct_valid=false;ct_coupon=false;if(ct_index){if(std::fclose(ct_index))++errors;ct_index=nullptr;}
 if(ct_blob){if(std::fclose(ct_blob))++errors;ct_blob=nullptr;}return ct_records;
}
