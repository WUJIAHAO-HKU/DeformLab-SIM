// Read-only, version-locked Cartesian articulation solver-state observations.
namespace {
FILE* aw_index=nullptr;FILE* aw_blob=nullptr;unsigned long long aw_records=0;
bool aw_selected(const char* name){
 return aw_index && (!std::strcmp(name,"artiPushImpulse") || !std::strcmp(name,"artiOutputVelocity") ||
 !std::strcmp(name,"stepArticulation1TTGS") || !std::strcmp(name,"artiSolveInternalConstraintsTGS1T") ||
 !std::strcmp(name,"artiPropagateRigidImpulsesAndSolveSelfConstraintsTGS1T") || !std::strcmp(name,"artiApplyTgsSubstepForces") ||
 !std::strcmp(name,"updateBodiesLaunch1T") || !std::strcmp(name,"updateBodiesLaunch_Part2") ||
 !std::strcmp(name,"updateBodiesLaunch_Part3") || !std::strcmp(name,"getArtiDofStates") ||
 !std::strcmp(name,"getArtiTransformStates"));
}
bool aw_snapshot(const char* name,void** params,const char* phase,unsigned long long seq){
 if(!aw_index || ct_step<0 || !enabled || !params)return false;
 CUcontext current=nullptr;if(cuCtxGetCurrent(&current)!=CUDA_SUCCESS || current!=context)return false;
 if(cuCtxSynchronize()!=CUDA_SUCCESS)return false;
 unsigned char core[344];CUdeviceptr pointer=*static_cast<CUdeviceptr*>(params[0]);
 if(cuMemcpyDtoH(core,pointer,sizeof(core))!=CUDA_SUCCESS)return false;
 unsigned count=field<unsigned>(core,44),links=field<unsigned>(core,248),dofs=field<unsigned>(core,252);
 // Scope is exactly the verified Cartesian fixture. Fail closed on layout drift.
 if(count!=1 || links!=4 || dofs!=3)return false;
 // Public state is an independent AoS, not a view of the solver blocks.
 // DWARF and getter PTX are pinned to the same native library as this hook.
 unsigned char articulation[496];
 if(cuMemcpyDtoH(articulation,field<CUdeviceptr>(core,32),sizeof(articulation))!=CUDA_SUCCESS)return false;
 if(field<unsigned short>(articulation,16)!=4 || field<unsigned short>(articulation,18)!=3)return false;
 std::fprintf(aw_index,"{\"seq\":%llu,\"step\":%d,\"kernel\":\"%s\",\"phase\":\"%s\",\"links\":%u,\"dofs\":%u,\"sections\":{",seq,ct_step.load(),name,phase,links,dofs);
 bool first=true,ok=true;
 auto save=[&](const char* key,const void* data,size_t size){
  long offset=std::ftell(aw_blob);if(offset<0 || (size && std::fwrite(data,1,size,aw_blob)!=size)){ok=false;return;}
  if(!first)std::fprintf(aw_index,",");first=false;
  std::fprintf(aw_index,"\"%s\":[%ld,%zu]",key,offset,size);
 };
 auto device=[&](const char* key,unsigned off,size_t size){
  std::vector<unsigned char> data(size);if(cuMemcpyDtoH(data.data(),field<CUdeviceptr>(core,off),size)!=CUDA_SUCCESS){ok=false;return;}
  save(key,data.data(),size);
 };
 save("core",core,sizeof(core));device("blocks",128,7168);
 device("link_blocks",136,size_t(links)*29184);device("dof_blocks",184,size_t(dofs)*12160);
 save("public_articulation",articulation,sizeof(articulation));
 auto public_device=[&](const char* key,unsigned off,size_t size){
  std::vector<unsigned char> data(size);
  if(cuMemcpyDtoH(data.data(),field<CUdeviceptr>(articulation,off),size)!=CUDA_SUCCESS){ok=false;return;}
  save(key,data.data(),size);
 };
 public_device("public_joint_positions",104,size_t(dofs)*4);
 public_device("public_joint_velocities",112,size_t(dofs)*4);
 public_device("public_joint_accelerations",120,size_t(dofs)*4);
 public_device("public_body_world",176,size_t(links)*28);
 public_device("public_body_actor",192,size_t(links)*28);
 std::fprintf(aw_index,"}}\n");++aw_records;return ok && !std::ferror(aw_index);
}
}
extern "C" int observer_articulation_work_start(const char* path){
 if(!path || !enabled || mode!=6 || !ct_coupon || aw_index)return -1;
 aw_index=std::fopen((std::string(path)+".jsonl").c_str(),"wx");
 aw_blob=std::fopen((std::string(path)+".bin").c_str(),"wx");aw_records=0;
 return aw_index && aw_blob?0:-2;
}
extern "C" unsigned long long observer_articulation_work_stop(){
 if(aw_index){if(std::fclose(aw_index))++errors;aw_index=nullptr;}
 if(aw_blob){if(std::fclose(aw_blob))++errors;aw_blob=nullptr;}return aw_records;
}
