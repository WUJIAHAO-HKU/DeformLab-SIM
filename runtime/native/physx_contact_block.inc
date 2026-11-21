// Synchronous, selected-frame research bridge; normal production path is off.
// It reuses the validated snapshot protocol and never advances time itself.
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/time.h>
#include <unistd.h>
namespace {
int block_policy=0,block_fd=-1;unsigned long long block_requests=0,block_updates=0;
std::vector<float> block_payload;unsigned block_contacts=0;bool block_pending=false;
std::vector<double> block_exact;std::vector<float4> block_old_p,block_old_d,block_old_v;std::vector<double3> block_old_low;
bool block_exact_pending=false;unsigned block_bodies=0;unsigned long long block_exact_calls=0;double block_h=0;
std::vector<double> block_angular;std::vector<float4> block_angular_before;CUdeviceptr block_pool=0;bool block_angular_pending=false;unsigned long long block_angular_calls=0;
bool block_io(void* data,size_t n,bool sending){auto p=static_cast<unsigned char*>(data);while(n){ssize_t k=sending?send(block_fd,p,n,MSG_NOSIGNAL):recv(block_fd,p,n,0);if(k<=0)return false;p+=k;n-=size_t(k);}return true;}
bool block_before(void** params,CUstream stream){
 if(!block_policy||ct_step<0)return true;
 if(block_exact_pending||block_angular_pending||!ct_valid||!ct_index||!ct_blob||!params||!stream||mode!=9||rr_policy!=2||friction_policy)return false;
 if(std::fflush(ct_index)||std::fflush(ct_blob))return false;
 unsigned long long request=ct_seq-1;unsigned header[2]={};
 if(!block_io(&request,8,true)||!block_io(header,8,false)||header[0]>1)return false;++block_requests;
 block_pending=header[0]==1;if(!block_pending)return true;if(block_policy<2)return false;
 unsigned c=0;if(cuMemcpyDtoH(&c,ct_args[3],4)!=CUDA_SUCCESS||c!=header[1]||c>16384)return false;
 block_contacts=c;block_payload.resize(size_t(c)*7+nodes*3);
 if(!block_io(block_payload.data(),block_payload.size()*4,false))return false;
 if(block_policy>=4){
  if(cuMemcpyDtoH(&block_bodies,ct_args[5]+268,4)!=CUDA_SUCCESS||block_bodies>64||!block_bodies)return false;
  block_exact.resize((nodes+block_bodies)*3);if(!block_io(block_exact.data(),block_exact.size()*8,false))return false;
  for(double v:block_exact)if(!std::isfinite(v))return false;
  block_h=ct_dt;
  if(block_policy>=5){
   unsigned char core[304],shared[224];
   if(cuMemcpyDtoH(core,ct_args[5],304)!=CUDA_SUCCESS||cuMemcpyDtoH(shared,ct_args[7],224)!=CUDA_SUCCESS)return false;
   unsigned offset=field<unsigned>(core,276);if(offset>4096)return false;
   block_pool=field<CUdeviceptr>(shared,160)+size_t(offset)*16;block_angular_before.resize(3*block_bodies);block_angular.resize(3*block_bodies);
   if(cuMemcpyDtoH(block_angular_before.data(),block_pool,block_angular_before.size()*16)!=CUDA_SUCCESS||!block_io(block_angular.data(),block_angular.size()*8,false))return false;
   for(double v:block_angular)if(!std::isfinite(v))return false;
   block_angular_pending=true;
  }
 }
 return true;
}
bool block_after(CUstream stream){
 if(!block_pending)return true;if(cuStreamSynchronize(stream)!=CUDA_SUCCESS)return false;
 unsigned char meta[432];if(cuMemcpyDtoH(meta,sb,432)!=CUDA_SUCCESS)return false;
 CUdeviceptr soft_delta=field<CUdeviceptr>(meta,272);std::vector<float4> rigid(2*block_contacts),soft(nodes);
 if(block_policy>=4){
  block_old_p.resize(nodes);block_old_d.resize(nodes);block_old_v.resize(nodes);block_old_low.resize(3*nodes);
  if(cuMemcpyDtoH(block_old_p.data(),positions,nodes*16)!=CUDA_SUCCESS||cuMemcpyDtoH(block_old_d.data(),field<CUdeviceptr>(meta,128),nodes*16)!=CUDA_SUCCESS||cuMemcpyDtoH(block_old_v.data(),velocities,nodes*16)!=CUDA_SUCCESS||cuMemcpyDtoH(block_old_low.data(),residual,nodes*72)!=CUDA_SUCCESS)return false;
  block_exact_pending=true;
 }
 if(cuMemcpyDtoH(rigid.data(),ct_args[8],rigid.size()*16)!=CUDA_SUCCESS||cuMemcpyDtoH(soft.data(),soft_delta,nodes*16)!=CUDA_SUCCESS)return false;
 size_t offset=0;for(auto& v:rigid){v.x=block_payload[offset++];v.y=block_payload[offset++];v.z=block_payload[offset++];}
 float* lambda=block_payload.data()+offset;offset+=block_contacts;
 for(auto& v:soft){v.x=block_payload[offset++];v.y=block_payload[offset++];v.z=block_payload[offset++];}
 if(cuMemcpyHtoD(ct_args[8],rigid.data(),rigid.size()*16)!=CUDA_SUCCESS||cuMemcpyHtoD(ct_args[9],lambda,block_contacts*4)!=CUDA_SUCCESS||cuMemcpyHtoD(soft_delta,soft.data(),nodes*16)!=CUDA_SUCCESS)return false;
 block_pending=false;++block_updates;return true;
}
bool block_after_capture(CUstream stream){
 if(!block_exact_pending)return true;if(cuStreamSynchronize(stream)!=CUDA_SUCCESS)return false;
 RigidReductionState states[64];if(cuMemcpyDtoH(states,rr_state,sizeof(states))!=CUDA_SUCCESS)return false;
 for(unsigned i=0;i<block_bodies;++i){double* dv=block_exact.data()+3*nodes+3*i;
  if(!states[i].pending){if(dv[0]!=0||dv[1]!=0||dv[2]!=0)return false;continue;}
  states[i].increment={dv[0],dv[1],dv[2]};
 }
 return cuMemcpyHtoD(rr_state,states,sizeof(states))==CUDA_SUCCESS;
}
bool block_after_reduce(CUstream stream){
 if(!block_angular_pending)return true;if(cuStreamSynchronize(stream)!=CUDA_SUCCESS)return false;
 std::vector<float4> pool(3*block_bodies);if(cuMemcpyDtoH(pool.data(),block_pool,pool.size()*16)!=CUDA_SUCCESS)return false;
 for(unsigned i=0;i<block_bodies;++i){
  auto before0=block_angular_before[i],before1=block_angular_before[block_bodies+i];auto dv=block_angular.data()+3*i;
  pool[i].w=float(double(before0.w)+dv[0]);pool[block_bodies+i].x=float(double(before1.x)+dv[1]);pool[block_bodies+i].y=float(double(before1.y)+dv[2]);
  if(!std::isfinite(pool[i].w)||!std::isfinite(pool[block_bodies+i].x)||!std::isfinite(pool[block_bodies+i].y))return false;
 }
 if(cuMemcpyHtoD(block_pool,pool.data(),pool.size()*16)!=CUDA_SUCCESS)return false;
 block_angular_pending=false;++block_angular_calls;return true;
}
bool block_after_apply(CUstream stream){
 if(!block_exact_pending)return true;if(cuStreamSynchronize(stream)!=CUDA_SUCCESS)return false;
 unsigned char meta[432];if(cuMemcpyDtoH(meta,sb,432)!=CUDA_SUCCESS)return false;
 auto advance=[](float4& high,double3& low,double x,double y,double z){
  double a=double(high.x)+low.x+x,b=double(high.y)+low.y+y,c=double(high.z)+low.z+z;
  high.x=float(a);high.y=float(b);high.z=float(c);low={a-high.x,b-high.y,c-high.z};
 };
 for(size_t i=0;i<nodes;++i){double* dv=block_exact.data()+3*i;
  advance(block_old_p[i],block_old_low[i],block_h*dv[0],block_h*dv[1],block_h*dv[2]);
  advance(block_old_d[i],block_old_low[nodes+i],block_h*dv[0],block_h*dv[1],block_h*dv[2]);
  advance(block_old_v[i],block_old_low[2*nodes+i],dv[0],dv[1],dv[2]);
 }
 if(cuMemcpyHtoD(positions,block_old_p.data(),nodes*16)!=CUDA_SUCCESS||cuMemcpyHtoD(field<CUdeviceptr>(meta,128),block_old_d.data(),nodes*16)!=CUDA_SUCCESS||cuMemcpyHtoD(velocities,block_old_v.data(),nodes*16)!=CUDA_SUCCESS||cuMemcpyHtoD(residual,block_old_low.data(),nodes*72)!=CUDA_SUCCESS)return false;
 block_exact_pending=false;++block_exact_calls;return true;
}
void block_stop(){if(block_exact_pending||block_angular_pending)++errors;block_angular_pending=block_exact_pending=block_pending=false;block_policy=0;if(block_fd>=0)close(block_fd);block_fd=-1;}
}
extern "C" int observer_contact_block_start(const char* path,int policy){
 if(!path||!ct_index||!enabled||mode!=9||rr_policy!=2||friction_policy||block_policy||policy<1||policy>6||std::strlen(path)>=sizeof(sockaddr_un::sun_path))return -1;
 block_fd=socket(AF_UNIX,SOCK_STREAM,0);if(block_fd<0)return -2;sockaddr_un address{};address.sun_family=AF_UNIX;std::strcpy(address.sun_path,path);
 timeval timeout{180,0};setsockopt(block_fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));setsockopt(block_fd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));
 if(connect(block_fd,reinterpret_cast<sockaddr*>(&address),sizeof(address))){block_stop();return -3;}
 block_policy=policy;return 0;
}
extern "C" unsigned long long observer_contact_block_count(int kind){return kind==0?block_requests:kind==1?block_updates:kind==2?block_exact_calls:block_angular_calls;}
