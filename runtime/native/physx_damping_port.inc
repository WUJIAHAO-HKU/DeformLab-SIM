// Read-only, synchronous final-contact port. Pinned Cartesian coupon only.
namespace {
bool dcp_enabled=false,dcp_pending=false;
unsigned long long dcp_frame=0,dcp_ready=0;
unsigned dcp_count=0;
double dcp_h=0.,dcp_rigid_delta=0.;
std::vector<double> dcp_rows,dcp_origin,dcp_output;
void dcp_reset(){
 dcp_enabled=dcp_pending=false;dcp_frame=dcp_ready=0;dcp_count=0;
 dcp_h=dcp_rigid_delta=0.;dcp_rows.clear();dcp_origin.clear();dcp_output.clear();
}
bool dcp_capture(CUstream stream,void** params){
 if(!dcp_enabled)return true;
 if(static_cast<unsigned char>(dc_values[3])!=1)return true;
 if(dcp_pending || !gap_complete() || gap_frame!=cp_frame || cp_frame!=dcp_ready+1 ||
    !params || !stream || cuStreamSynchronize(stream)!=CUDA_SUCCESS)return false;
 unsigned bad=0,count=0;unsigned char soft[432],core[48];
 if(cuMemcpyDtoH(&bad,rejected,4)!=CUDA_SUCCESS || bad ||
    cuMemcpyDtoH(&count,*static_cast<CUdeviceptr*>(params[3]),4)!=CUDA_SUCCESS || count>1024 ||
    cuMemcpyDtoH(soft,sb,sizeof(soft))!=CUDA_SUCCESS || field<unsigned>(soft,372)!=nodes ||
    cuMemcpyDtoH(core,*static_cast<CUdeviceptr*>(params[5]),sizeof(core))!=CUDA_SUCCESS)return false;
 CUdeviceptr dp=field<CUdeviceptr>(soft,128),rv=field<CUdeviceptr>(core,40);
 if(!dp || !rv || !residual)return false;
 std::vector<float4> delta(nodes);std::vector<double> debt(nodes*3);
 float4 rigid;
 dcp_rows.resize(count*6);dcp_origin.resize(nodes);
 if((count && cuMemcpyDtoH(dcp_rows.data(),nr_rows,count*6*sizeof(double))!=CUDA_SUCCESS) ||
    cuMemcpyDtoH(delta.data(),dp,nodes*sizeof(float4))!=CUDA_SUCCESS ||
    cuMemcpyDtoH(debt.data(),residual+nodes*3*sizeof(double),nodes*3*sizeof(double))!=CUDA_SUCCESS ||
    cuMemcpyDtoH(&rigid,rv+11*sizeof(float4),sizeof(rigid))!=CUDA_SUCCESS)return false;
 for(size_t i=0;i<nodes;++i){dcp_origin[i]=double(delta[i].z)+debt[3*i+2];if(!std::isfinite(dcp_origin[i]))return false;}
 dcp_rigid_delta=rigid.z;dcp_h=*static_cast<float*>(params[11]);
 if(!std::isfinite(dcp_rigid_delta)||!std::isfinite(dcp_h)||dcp_h<=0)return false;
 dcp_count=count;dcp_frame=cp_frame;dcp_pending=true;return true;
}
bool dcp_finish(CUstream stream,CUdeviceptr links){
 if(!dcp_enabled)return true;
 if(!dcp_pending || dcp_frame!=cp_frame || !links || !stream ||
    cuStreamSynchronize(stream)!=CUDA_SUCCESS)return false;
 unsigned bad=0;unsigned char soft[432];
 if(cuMemcpyDtoH(&bad,rejected,4)!=CUDA_SUCCESS || bad ||
    cuMemcpyDtoH(soft,sb,sizeof(soft))!=CUDA_SUCCESS)return false;
 CUdeviceptr dp=field<CUdeviceptr>(soft,128),vp=field<CUdeviceptr>(soft,88);
 if(!dp || !vp)return false;
 std::vector<float4> delta(nodes),velocity(nodes);std::vector<double> debt(nodes*3);
 float vr,rd;
 // Layout proof: block size 29184, mMotionVelocity 9472, mDeltaMotion 11008;
 // z is float offset 129 in the packed SpatialVectorF, NOT byte offset 128.
 if(cuMemcpyDtoH(&vr,links+3*29184+9472+129*4,4)!=CUDA_SUCCESS ||
    cuMemcpyDtoH(&rd,links+3*29184+11008+129*4,4)!=CUDA_SUCCESS ||
    cuMemcpyDtoH(delta.data(),dp,nodes*sizeof(float4))!=CUDA_SUCCESS ||
    cuMemcpyDtoH(velocity.data(),vp,nodes*sizeof(float4))!=CUDA_SUCCESS ||
    cuMemcpyDtoH(debt.data(),residual+nodes*3*sizeof(double),nodes*3*sizeof(double))!=CUDA_SUCCESS ||
    !std::isfinite(vr) || !std::isfinite(rd))return false;
 std::vector<double> output;
 for(unsigned i=0;i<dcp_count;++i){
  const double* r=dcp_rows.data()+6*i;if(r[0]==-1.)continue;
  for(unsigned j=0;j<6;++j)if(!std::isfinite(r[j]))return false;
  if(r[0]<0 || r[0]>=nodes || r[0]!=double(unsigned(r[0])) || (r[1]!=0.&&r[1]!=1.))return false;
  unsigned node=unsigned(r[0]);double sign=r[1]?1.:-1.;
  double ds=double(delta[node].z)+debt[3*node+2]-dcp_origin[node];
  double gap=r[4]+sign*((double(rd)-dcp_rigid_delta)*r[1]-ds);
  double relative=sign*(double(vr)*r[1]-double(velocity[node].z));
  if(!std::isfinite(gap)||!std::isfinite(relative))return false;
  for(double value:{r[0],r[1],gap,relative,double(velocity[node].z),double(vr)})output.push_back(value);
 }
 dcp_output.swap(output);dcp_pending=false;dcp_ready=dcp_frame;return true;
}
}
extern "C" int observer_damping_port_start(){
 std::lock_guard<std::mutex> guard(lock);
 if(!enabled || !gap_active() || dcp_enabled || gap_prepares || sb || stream_seen)return -1;
 dcp_reset();dcp_enabled=true;return 0;
}
extern "C" int observer_damping_port_read(unsigned long long frame,double* out,unsigned capacity,double* meta){
 std::lock_guard<std::mutex> guard(lock);
 if(!enabled || !dcp_enabled)return -1;
 if(!out || !meta || capacity!=1024 || !frame)return -2;
 if(dcp_pending || dcp_ready!=frame || dcp_frame!=frame)return -3;
 if(!dcp_output.empty())std::memcpy(out,dcp_output.data(),dcp_output.size()*sizeof(double));
 meta[0]=double(dcp_output.size()/6);meta[1]=dcp_h;meta[2]=double(nodes);meta[3]=double(frame);
 return 0;
}
