// Version-locked, synchronous velocity-only transaction at a completed frame.
// No joint setters/kinematic refresh: preserve poses, targets and force history.
namespace {
struct DcvBuffer {CUdeviceptr pointer;std::vector<unsigned char> before,after;};
float dcv_float(const std::vector<unsigned char>& b,size_t offset){float v;std::memcpy(&v,b.data()+offset,4);return v;}
void dcv_put(std::vector<unsigned char>& b,size_t offset,float v){std::memcpy(b.data()+offset,&v,4);}
}
extern "C" int observer_damping_velocity_publish(unsigned long long frame,
 const float* old_joint,const float* new_joint,const float* old_link,const float* new_link,
 const float* old_soft,const float* new_soft,unsigned n){
 std::lock_guard<std::mutex> guard(lock);
 if(!enabled || !dcp_enabled || dcp_pending || dc_pending || dc_have_drive || !gap_complete() || !frame || frame!=dcp_ready || frame!=cp_frame ||
    frame<=dcv_last_written || !dcv_core || !dcv_solver_core || !context)return -1;
 if(!old_joint||!new_joint||!old_link||!new_link||!old_soft||!new_soft||n!=nodes||!n||n>4096)return -2;
 for(unsigned i=0;i<3;++i)if(!std::isfinite(old_joint[i])||!std::isfinite(new_joint[i]))return -2;
 for(unsigned i=0;i<12;++i)if(!std::isfinite(old_link[i])||!std::isfinite(new_link[i]))return -2;
 for(unsigned i=0;i<3*n;++i)if(!std::isfinite(old_soft[i])||!std::isfinite(new_soft[i]))return -2;
 for(unsigned i=0;i<3;++i)if(old_link[i]!=0.f||new_link[i]!=0.f)return -2;
 CUresult c=cuCtxPushCurrent(context);if(c!=CUDA_SUCCESS)return int(c);
 auto finish=[](int code){CUcontext old;CUresult c=cuCtxPopCurrent(&old);return code?code:int(c);};
 if(cuCtxSynchronize()!=CUDA_SUCCESS)return finish(-3);
 unsigned char core[344],arti[496],solver[48],soft[432];unsigned bad=0;
 if(cuMemcpyDtoH(core,dcv_core,sizeof(core))!=CUDA_SUCCESS ||
    field<unsigned>(core,44)!=1 || field<unsigned>(core,248)!=4 || field<unsigned>(core,252)!=3)return finish(-4);
 if(cuMemcpyDtoH(arti,field<CUdeviceptr>(core,32),sizeof(arti))!=CUDA_SUCCESS ||
    field<unsigned short>(arti,16)!=4 || field<unsigned short>(arti,18)!=3 ||
    cuMemcpyDtoH(solver,dcv_solver_core,sizeof(solver))!=CUDA_SUCCESS ||
    cuMemcpyDtoH(soft,sb,sizeof(soft))!=CUDA_SUCCESS || field<unsigned>(soft,372)!=n ||
    field<CUdeviceptr>(soft,88)!=velocities ||
    cuMemcpyDtoH(&bad,rejected,4)!=CUDA_SUCCESS || bad)return finish(-4);
 std::vector<DcvBuffer> buffers;buffers.reserve(6);
 auto add=[&](CUdeviceptr p,size_t size){
  if(!p || p+size<p)return false;
  for(const auto& b:buffers)if(p<b.pointer+b.before.size() && b.pointer<p+size)return false;
  DcvBuffer b{p,std::vector<unsigned char>(size),{}};
  if(cuMemcpyDtoH(b.before.data(),p,size)!=CUDA_SUCCESS)return false;
  b.after=b.before;buffers.push_back(std::move(b));return true;
 };
 // Matched DWARF: public motionVelocities is 24-byte UnAlignedSpatialVector,
 // angular top at 0 and linear bottom at 12. Solver blocks use packed lanes.
 if(!add(field<CUdeviceptr>(arti,112),12) || !add(field<CUdeviceptr>(arti,56),4*24) ||
    !add(field<CUdeviceptr>(core,184),3*12160) || !add(field<CUdeviceptr>(core,136),4*29184) ||
    !add(field<CUdeviceptr>(solver,40),16*16) || !add(velocities,n*16))return finish(-5);
 for(unsigned i=0;i<3;++i){
  size_t off=i*12160+128;
  if(dcv_float(buffers[0].before,i*4)!=old_joint[i] || dcv_float(buffers[2].before,off)!=old_joint[i])return finish(-6);
  dcv_put(buffers[0].after,i*4,new_joint[i]);dcv_put(buffers[2].after,off,new_joint[i]);
 }
 const size_t linear[3]={12,512,516};
 for(unsigned i=0;i<4;++i){
  for(unsigned j=0;j<3;++j){
   size_t pub=i*24+12+j*4,block=i*29184+9472+linear[j];
   if(dcv_float(buffers[1].before,i*24+j*4)!=0.f || dcv_float(buffers[3].before,i*29184+9472+j*4)!=0.f ||
      dcv_float(buffers[1].before,pub)!=old_link[3*i+j] || dcv_float(buffers[3].before,block)!=old_link[3*i+j])return finish(-7);
   dcv_put(buffers[1].after,pub,new_link[3*i+j]);dcv_put(buffers[3].after,block,new_link[3*i+j]);
   dcv_put(buffers[4].after,i*16+j*4,new_link[3*i+j]);
  }
 }
 for(unsigned i=0;i<n;++i){
  if(!std::isfinite(dcv_float(buffers[5].before,i*16+12)) || dcv_float(buffers[5].before,i*16+12)<=0.f)return finish(-8);
  for(unsigned j=0;j<3;++j){
   size_t off=i*16+j*4;if(dcv_float(buffers[5].before,off)!=old_soft[3*i+j])return finish(-8);
   dcv_put(buffers[5].after,off,new_soft[3*i+j]);
  }
 }
 // No simulation is launched between these synchronous copies. Verify every
 // written byte; on a driver failure restore the complete before-images.
 bool ok=true;
 for(auto& b:buffers)if(cuMemcpyHtoD(b.pointer,b.after.data(),b.after.size())!=CUDA_SUCCESS){ok=false;break;}
 if(ok)for(auto& b:buffers){std::vector<unsigned char> check(b.after.size());
  if(cuMemcpyDtoH(check.data(),b.pointer,check.size())!=CUDA_SUCCESS || check!=b.after){ok=false;break;}}
 if(!ok){bool restored=true;for(auto& b:buffers){
   if(cuMemcpyHtoD(b.pointer,b.before.data(),b.before.size())!=CUDA_SUCCESS)restored=false;
  }++errors;return finish(restored?-9:-10);}
 dcv_last_written=frame;++dcv_writes;return finish(0);
}
extern "C" unsigned long long observer_damping_velocity_count(){return dcv_writes;}
