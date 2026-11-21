// Opt-in observation and round-trip correction of pinned TGS angular coordinates.
// Does not change the native unconstrained/gyroscopic world velocity target.
namespace {
int at_policy=0;FILE* at_index=nullptr;FILE* at_blob=nullptr;
unsigned long long at_records=0,at_pre_calls=0,at_changed=0,at_kept=0;
bool at_selected(const char* n){return !std::strcmp(n,"preIntegrationLaunchTGS")||!std::strcmp(n,"initializeSolverVelocitiesTGS")||!std::strcmp(n,"integrateCoreParallelLaunchTGS")||!std::strcmp(n,"writeBackBodiesTGS")||!std::strcmp(n,"computeAverageSolverBodyVelocityTGS")||!std::strcmp(n,"propagateAverageSolverBodyVelocityTGS");}
bool at_active(const char* n){return at_policy&&ct_step.load()>=0&&at_selected(n);}
bool at_snapshot(const char* name,void** params,const char* phase,unsigned long long seq){
 if(!at_active(name))return true;if(!params||cuCtxSynchronize()!=CUDA_SUCCESS)return false;
 bool pre=!std::strcmp(name,"preIntegrationLaunchTGS");unsigned nb=0,offset=0,pool_offset=0;float dt=0;
 CUdeviceptr data=0,txi=0,out=0,pose=0,sims=0,pool=0,motion=0;std::vector<unsigned long long> ids;std::vector<unsigned char> body_data;
 if(pre){
  offset=*static_cast<unsigned*>(params[0]);nb=*static_cast<unsigned*>(params[1]);dt=*static_cast<float*>(params[2]);
  data=*static_cast<CUdeviceptr*>(params[4]);txi=*static_cast<CUdeviceptr*>(params[6]);sims=*static_cast<CUdeviceptr*>(params[7]);pose=*static_cast<CUdeviceptr*>(params[9]);out=*static_cast<CUdeviceptr*>(params[10]);
  if(!nb||nb>64||offset>nb)return false;ids.resize(nb);
  if(cuMemcpyDtoH(ids.data(),*static_cast<CUdeviceptr*>(params[8]),nb*8)!=CUDA_SUCCESS)return false;
 }else{
  bool integrate=!std::strcmp(name,"integrateCoreParallelLaunchTGS");unsigned char core[304],shared[224];
  if(cuMemcpyDtoH(core,*static_cast<CUdeviceptr*>(params[integrate?1:0]),304)!=CUDA_SUCCESS||cuMemcpyDtoH(shared,*static_cast<CUdeviceptr*>(params[integrate?2:1]),224)!=CUDA_SUCCESS)return false;
  nb=field<unsigned>(core,268);pool_offset=field<unsigned>(core,276);if(!nb||nb>64||pool_offset>4096)return false;
  data=field<CUdeviceptr>(core,16);txi=field<CUdeviceptr>(core,24);out=field<CUdeviceptr>(core,0);pose=field<CUdeviceptr>(core,8);sims=field<CUdeviceptr>(core,240);pool=field<CUdeviceptr>(shared,160);motion=field<CUdeviceptr>(core,56);
 }
 body_data.resize(nb*96);if(cuMemcpyDtoH(body_data.data(),data,body_data.size())!=CUDA_SUCCESS)return false;
 if(!pre){ids.resize(nb);for(unsigned i=0;i<nb;++i)ids[i]=field<unsigned long long>(body_data.data()+i*96,64);}
 std::vector<unsigned char> body_sim(nb*224,0);std::vector<unsigned> present(nb,0);
 for(unsigned i=offset;i<nb;++i){
  if(!pre&&!(field<float>(body_data.data()+i*96,28)>0))continue;
  unsigned id=unsigned(ids[i]);if(id>1000000)return false;
  if(cuMemcpyDtoH(body_sim.data()+i*224,sims+size_t(id)*224,224)!=CUDA_SUCCESS)return false;present[i]=1;
 }
 std::fprintf(at_index,"{\"seq\":%llu,\"step\":%d,\"kernel\":\"%s\",\"phase\":\"%s\",\"bodies\":%u,\"offset\":%u,\"dt\":%.17g,\"pool_offset\":%u,\"sections\":{",seq,ct_step.load(),name,phase,nb,offset,double(dt),pool_offset);
 bool first=true,ok=true;
 auto host=[&](const char* key,const void* p,size_t size){if(!ok)return;long start=std::ftell(at_blob);if(start<0||(size&&std::fwrite(p,1,size,at_blob)!=size)){ok=false;return;}std::fprintf(at_index,"%s\"%s\":[%ld,%zu]",first?"":",",key,start,size);first=false;};
 auto device=[&](const char* key,CUdeviceptr p,size_t size){std::vector<unsigned char> buf(size);if(size&&cuMemcpyDtoH(buf.data(),p,size)!=CUDA_SUCCESS){ok=false;return;}host(key,buf.data(),size);};
 host("ids",ids.data(),nb*8);host("present",present.data(),nb*4);host("body_data",body_data.data(),body_data.size());host("body_sim",body_sim.data(),body_sim.size());
 device("txi",txi,nb*64);device("out_velocity",out,nb*32);device("out_pose",pose,nb*32);
 if(pool)device("body_pool",pool,size_t(pool_offset+3*nb)*16);
 if(motion)device("motion",motion,nb*32);
 std::fprintf(at_index,"}}\n");if(!ok||std::ferror(at_index))return false;++at_records;return true;
}
bool at_solve3(const double s[3][3],const double rhs[3],double x[3],bool transpose=false){
 double a[3][4];for(int i=0;i<3;++i){for(int j=0;j<3;++j)a[i][j]=transpose?s[j][i]:s[i][j];a[i][3]=rhs[i];}
 for(int k=0;k<3;++k){int pivot=k;for(int i=k+1;i<3;++i)if(std::fabs(a[i][k])>std::fabs(a[pivot][k]))pivot=i;
  if(!std::isfinite(a[pivot][k])||a[pivot][k]==0)return false;
  for(int j=k;j<4;++j)std::swap(a[k][j],a[pivot][j]);
  for(int i=k+1;i<3;++i){double f=a[i][k]/a[k][k];for(int j=k;j<4;++j)a[i][j]-=f*a[k][j];}
 }
 for(int i=2;i>=0;--i){double v=a[i][3];for(int j=i+1;j<3;++j)v-=a[i][j]*x[j];x[i]=v/a[i][i];if(!std::isfinite(x[i]))return false;}return true;
}
bool at_inverse(void** params){
 if(cuCtxSynchronize()!=CUDA_SUCCESS)return false;
 unsigned offset=*static_cast<unsigned*>(params[0]),nb=*static_cast<unsigned*>(params[1]);if(!nb||nb>64||offset>nb)return false;
 auto data=*static_cast<CUdeviceptr*>(params[4]),txi=*static_cast<CUdeviceptr*>(params[6]),out=*static_cast<CUdeviceptr*>(params[10]);
 std::vector<float> d(nb*24),t(nb*16);std::vector<float4> angular(nb);
 if(cuMemcpyDtoH(d.data(),data,nb*96)!=CUDA_SUCCESS||cuMemcpyDtoH(t.data(),txi,nb*64)!=CUDA_SUCCESS||cuMemcpyDtoH(angular.data(),out+nb*16,nb*16)!=CUDA_SUCCESS)return false;
 ++at_pre_calls;
 for(unsigned b=offset;b<nb;++b){if(!(d[24*b+7]>0))continue;
  double s[3][3],omega[3],ideal[3]={},old_error[3],new_error[3],old_l[3]={},new_l[3]={};float candidate[3];
  float* current=&angular[b].x;
  for(int i=0;i<3;++i){omega[i]=d[24*b+i];for(int j=0;j<3;++j)s[i][j]=t[16*b+7+3*j+i];}
  if(!at_solve3(s,omega,ideal))return false;
  for(int i=0;i<3;++i){candidate[i]=float(ideal[i]);old_error[i]=double(current[i])-ideal[i];new_error[i]=double(candidate[i])-ideal[i];}
  if(!at_solve3(s,old_error,old_l,true)||!at_solve3(s,new_error,new_l,true))return false;
  double old_norm=0,new_norm=0;bool changed=false;
  for(int i=0;i<3;++i){old_norm+=old_l[i]*old_l[i];new_norm+=new_l[i]*new_l[i];changed|=candidate[i]!=current[i];}
  // Preserve the native result if this float candidate is worse in angular
  // momentum coordinates. No global-state or measured-system correction.
  if(changed&&new_norm<old_norm){for(int i=0;i<3;++i)current[i]=candidate[i];++at_changed;}else ++at_kept;
 }
 return cuMemcpyHtoD(out+nb*16,angular.data(),nb*16)==CUDA_SUCCESS;
}
bool at_after(const char* name,void** params,unsigned long long seq){
 if(at_policy!=2||ct_step.load()<0||std::strcmp(name,"preIntegrationLaunchTGS"))return true;
 return at_snapshot(name,params,"native",seq)&&at_inverse(params);
}
}
extern "C" int observer_angular_transport_start(const char* path,int policy){
 if(!path||!ct_index||!enabled||mode!=9||at_policy||policy<1||policy>2)return -1;
 at_index=std::fopen((std::string(path)+".jsonl").c_str(),"wx");at_blob=std::fopen((std::string(path)+".bin").c_str(),"wx");if(!at_index||!at_blob)return -2;at_policy=policy;return 0;
}
extern "C" unsigned long long observer_angular_transport_count(int kind){return kind==0?at_records:kind==1?at_pre_calls:kind==2?at_changed:at_kept;}
extern "C" void observer_angular_transport_stop(){at_policy=0;if(at_index){if(std::fclose(at_index))++errors;at_index=nullptr;}if(at_blob){if(std::fclose(at_blob))++errors;at_blob=nullptr;}}
