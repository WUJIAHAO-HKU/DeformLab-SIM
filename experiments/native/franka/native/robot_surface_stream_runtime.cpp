// Stream-ordered bridge: GPU validation per round, explicit host barriers at
// frame boundaries and diagnostic outputs. Errors are sticky, never fallback.
#include <cuda.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include "robot_surface_stream_state.h"
struct Row {unsigned node,tet,corner,source;};
struct Runtime {
 CUcontext context=nullptr;CUmodule module=nullptr;CUstream stream=nullptr;
 CUfunction kernels[5]={};SurfaceStreamState state={};CUdeviceptr device_state=0;
 unsigned frame=0,round=0,phase=0;bool begun=false,stream_bound=false,bound=false,topology_bound=false;
 int error=0;unsigned long long bytes=0,metrics[5]={};CUdeviceptr reaction=0;unsigned reaction_capacity=0;
 struct Allocation {CUdeviceptr ptr;size_t size;};std::vector<Allocation> allocations;
 int fail(int code){if(!error)error=code;return error;}
 int check(){CUcontext c=nullptr;if(error)return error;if(cuCtxGetCurrent(&c)!=CUDA_SUCCESS||c!=context)return fail(801);return 0;}
 bool allocate(uint64_t& p,size_t size){CUdeviceptr q=0;if(cuMemAlloc(&q,size+64)!=CUDA_SUCCESS)return false;p=q;allocations.push_back({q,size});bytes+=size+64;return cuMemsetD8(q,0,size)==CUDA_SUCCESS&&cuMemsetD8(q+size,0xa5,64)==CUDA_SUCCESS;}
 bool pointer(CUdeviceptr p,size_t size){CUcontext c=nullptr;CUdeviceptr base=0;size_t total=0;return p&&cuPointerGetAttribute(&c,CU_POINTER_ATTRIBUTE_CONTEXT,p)==CUDA_SUCCESS&&c==context&&cuMemGetAddressRange(&base,&total,p)==CUDA_SUCCESS&&p>=base&&size<=total&&p-base<=total-size;}
 size_t extent(CUdeviceptr p){
  CUcontext c=nullptr;CUdeviceptr base=0;size_t total=0;
  if(!p||cuPointerGetAttribute(&c,CU_POINTER_ATTRIBUTE_CONTEXT,p)!=CUDA_SUCCESS||c!=context||cuMemGetAddressRange(&base,&total,p)!=CUDA_SUCCESS||p<base||p-base>total)return 0;
  return total-(p-base);
 }
 int sync(){auto c=cuStreamSynchronize(stream);return c==CUDA_SUCCESS?0:fail(11000+int(c));}
 int launch(unsigned i,void** args){auto c=cuLaunchKernel(kernels[i],1,1,1,128,1,1,0,stream,args,nullptr);return c==CUDA_SUCCESS?0:fail(10000+int(c));}
 int upload(){if(sync())return error;return cuMemcpyHtoD(device_state,&state,sizeof(state))==CUDA_SUCCESS?0:fail(807);}
 int barrier(){
  if(sync())return error;
  if(cuMemcpyDtoH(metrics,state.telemetry,sizeof(metrics))!=CUDA_SUCCESS)return fail(805);
  if(metrics[4])return fail(int(metrics[4]));
  unsigned h[8];if(cuMemcpyDtoH(h,state.header,32)!=CUDA_SUCCESS)return fail(805);
  if(h[6])return fail(20000+int(h[6]));
  if(begun&&(h[0]!=0x53555246u||h[1]!=frame||h[2]!=round||h[3]!=(phase?2u:1u)||h[4]!=state.nodes||h[5]!=state.boundary_count))return fail(806);
  return error;
 }
 void release(){for(auto a:allocations)cuMemFree(a.ptr);allocations.clear();if(module)cuModuleUnload(module);module=nullptr;}
};
extern "C" void* robot_surface_runtime_create(const char* ptx,unsigned nodes,unsigned tets,const Row* boundary,unsigned count,double cx,double cy,double hx,double hy,double plane){
 if(!ptx||!*ptx||!nodes||nodes>65536||!tets||tets>1048576||!count||count>nodes||!boundary||!std::isfinite(cx)||!std::isfinite(cy)||!std::isfinite(hx)||!std::isfinite(hy)||!std::isfinite(plane)||hx<=0||hy<=0)return nullptr;
 for(unsigned i=0;i<count;++i)if(boundary[i].node>=nodes||boundary[i].tet>=tets||boundary[i].corner>=4||(i&&boundary[i].node<=boundary[i-1].node))return nullptr;
 auto r=new Runtime;auto& s=r->state;s.nodes=nodes;s.tets=tets;s.boundary_count=count;s.capacity=16384+count;s.cx=cx;s.cy=cy;s.hx=hx;s.hy=hy;s.plane=plane;s.epsilon=2e-7;
 if(cuCtxGetCurrent(&r->context)!=CUDA_SUCCESS||!r->context||cuModuleLoad(&r->module,ptx)!=CUDA_SUCCESS){r->release();delete r;return nullptr;}
 const char* names[]={"surface_stream_reset","surface_stream_prepare","surface_stream_after_query","surface_stream_bind_reaction","surface_stream_publish"};
 for(unsigned i=0;i<5;++i)if(cuModuleGetFunction(&r->kernels[i],r->module,names[i])!=CUDA_SUCCESS){r->release();delete r;return nullptr;}
 unsigned cap=s.capacity;uint64_t packet=0;
 bool ok=r->allocate(s.epoch_count,4)&&r->allocate(s.header,32)&&r->allocate(s.cache,nodes*4)&&r->allocate(s.seen,nodes*4)&&r->allocate(s.active,nodes*4)&&r->allocate(s.boundary,count*16)&&r->allocate(s.plan,16)&&r->allocate(s.selected,count*16)&&r->allocate(s.zero,cap*32)&&r->allocate(s.info,cap*16)&&r->allocate(s.sid,cap*4)&&r->allocate(s.constraints,((cap+31)/32)*3200)&&r->allocate(s.lambda,cap*4)&&r->allocate(s.rigid,cap*32)&&r->allocate(s.count,4)&&r->allocate(s.fixed_topology,tets*16)&&r->allocate(s.frame_info,16384*16)&&r->allocate(s.frame_ids,16384*4)&&r->allocate(s.telemetry,40)&&r->allocate(packet,sizeof(s));
 r->device_state=packet;
 // Include the descriptor buffer's own guard as well.
 unsigned guards=unsigned(r->allocations.size())+1;
 ok=ok&&r->allocate(s.guards,guards*8);s.guard_count=unsigned(r->allocations.size());
 std::vector<uint64_t> tails;for(auto a:r->allocations)tails.push_back(a.ptr+a.size);
 if(!ok||s.guard_count!=guards||cuMemcpyHtoD(s.guards,tails.data(),tails.size()*8)!=CUDA_SUCCESS||cuMemcpyHtoD(s.boundary,boundary,count*16)!=CUDA_SUCCESS||r->upload()){r->release();delete r;return nullptr;}
 return r;
}
extern "C" int robot_surface_runtime_begin(void* handle,unsigned frame){
 auto r=static_cast<Runtime*>(handle);if(!r)return -1;if(r->check())return r->error;
 if(r->phase||(r->begun&&frame<=r->frame))return r->fail(810);
 if(r->barrier())return r->error;
 r->frame=frame;r->round=0;r->begun=true;r->bound=false;r->reaction=0;r->state.frame=frame;
 if(r->upload())return r->error;void* args[]={&r->device_state,&r->frame};
 if(r->launch(0,args))return r->error;return r->barrier();
}
// Bind the finite native ABI once per frame; GPU checks its values every round.
extern "C" int robot_surface_runtime_native_bind(void* handle,const CUdeviceptr* p,unsigned map_count){
 auto r=static_cast<Runtime*>(handle);if(!r||!p)return -1;if(r->check())return r->error;
 if(!r->begun||r->phase||r->round||r->bound||map_count>65536)return r->fail(831);
 size_t sizes[]={432,280,304,252,size_t(r->state.nodes)*16,size_t(map_count)*4};
 for(unsigned i=0;i<6;++i)if((i!=5||map_count)&&!r->pointer(p[i],sizes[i]))return r->fail(832);
 for(unsigned i=0;i<6;++i)r->state.native_binding[i]=p[i];r->state.native_validation=1;r->state.map_count=map_count;return 0;
}
extern "C" int robot_surface_runtime_prepare(void* handle,const CUdeviceptr* input,unsigned frame,unsigned round,CUstream stream){
 auto r=static_cast<Runtime*>(handle);if(!r||!input)return -1;if(r->check())return r->error;
 if(!r->begun||r->phase||frame!=r->frame||round!=r->round||input[8])return r->fail(811);
 CUstreamCaptureStatus capture;if(cuStreamIsCapturing(stream,&capture)!=CUDA_SUCCESS||capture!=CU_STREAM_CAPTURE_STATUS_NONE)return r->fail(812);
 if(r->stream_bound&&stream!=r->stream)return r->fail(813);r->stream=stream;r->stream_bound=true;
 bool changed=false;for(unsigned i=0;i<8;++i)changed=changed||r->state.input[i]!=input[i];
 if(!r->bound||changed){
  if(r->barrier()||!r->pointer(input[4],4))return r->fail(814);
  // Never read the transient native count on the CPU. Validate allocation
  // capacities here, then bind the count and IDs on the consumer GPU stream.
  size_t strides[]={16,4,3200,4};
  for(unsigned i=0;i<4;++i){
   size_t bytes=r->extent(input[i]);if(!bytes)return r->fail(816);
   size_t capacity=(bytes/strides[i])*(i==2?32:1);
   r->state.input_capacity[i]=unsigned(capacity<16384?capacity:16384);
  }
  if(!r->pointer(input[5],size_t(r->state.tets)*16)||!r->pointer(input[6],size_t(r->state.nodes)*16)||!r->pointer(input[7],size_t(r->state.nodes)*16))return r->fail(816);
  for(unsigned i=0;i<8;++i)r->state.input[i]=input[i];r->state.input[8]=0;
  if(!r->topology_bound){if(cuMemcpyDtoDAsync(r->state.fixed_topology,input[5],size_t(r->state.tets)*16,stream)!=CUDA_SUCCESS)return r->fail(826);r->topology_bound=true;}
  if(r->upload())return r->error;r->bound=true;
 }
 void* args[]={&r->device_state,&r->round};if(r->launch(1,args))return r->error;r->phase=1;return 0;
}
// Diagnostic barrier: returns actual count only after GPU preparation is valid.
extern "C" int robot_surface_runtime_outputs(void* handle,CUdeviceptr* pointers,unsigned* counts){
 auto r=static_cast<Runtime*>(handle);if(!r||!pointers||!counts)return -1;if(r->check())return r->error;if(!r->phase)return r->fail(818);
 if(r->barrier())return r->error;auto& s=r->state;CUdeviceptr p[]={s.info,s.sid,s.constraints,s.lambda,s.rigid,s.count};std::memcpy(pointers,p,sizeof(p));
 return cuMemcpyDtoH(counts,s.plan,16)==CUDA_SUCCESS?0:r->fail(817);
}
// Live launch bound, NOT actual selected count. Native kernels read s.count.
extern "C" int robot_surface_runtime_launch_outputs(void* handle,CUdeviceptr* pointers,unsigned* counts){
 auto r=static_cast<Runtime*>(handle);if(!r||!pointers||!counts)return -1;if(r->check())return r->error;if(!r->phase)return r->fail(818);
 auto& s=r->state;CUdeviceptr p[]={s.info,s.sid,s.constraints,s.lambda,s.rigid,s.count};std::memcpy(pointers,p,sizeof(p));unsigned c[]={0,~0u,s.capacity,~0u};std::memcpy(counts,c,sizeof(c));return 0;
}
extern "C" int robot_surface_runtime_after_query(void* handle){
 auto r=static_cast<Runtime*>(handle);if(!r)return -1;if(r->check())return r->error;if(r->phase!=1)return r->fail(819);
 void* args[]={&r->device_state};if(r->launch(2,args))return r->error;r->phase=2;return 0;
}
extern "C" int robot_surface_runtime_bind_reaction(void* handle,CUdeviceptr original){
 auto r=static_cast<Runtime*>(handle);if(!r)return -1;if(r->check())return r->error;if(r->phase!=2)return r->fail(820);
 if(r->reaction!=original){size_t capacity=r->extent(original)/32;if(!capacity)return r->fail(820);r->reaction=original;r->reaction_capacity=unsigned(capacity<16384?capacity:16384);}
 uint64_t source=original;void* args[]={&r->device_state,&source,&r->reaction_capacity};if(r->launch(3,args))return r->error;r->phase=3;return 0;
}
extern "C" int robot_surface_runtime_publish(void* handle){
 auto r=static_cast<Runtime*>(handle);if(!r)return -1;if(r->check())return r->error;if(r->phase!=3)return r->fail(821);
 void* args[]={&r->device_state};if(r->launch(4,args))return r->error;++r->round;r->phase=0;return 0;
}
extern "C" int robot_surface_runtime_stats(void* handle,unsigned long long* values){
 auto r=static_cast<Runtime*>(handle);if(!r||!values)return -1;
 CUcontext old=nullptr;bool push=cuCtxGetCurrent(&old)!=CUDA_SUCCESS||old!=r->context;
 if(push&&cuCtxPushCurrent(r->context)!=CUDA_SUCCESS)return r->fail(824);
 r->barrier();unsigned long long v[]={r->metrics[0],r->metrics[1],r->metrics[2],r->metrics[3],r->bytes,r->phase,unsigned(r->error),r->frame,r->round};std::memcpy(values,v,sizeof(v));
 if(push)cuCtxPopCurrent(&old);return r->error;
}
extern "C" int robot_surface_runtime_stop(void* handle){
 auto r=static_cast<Runtime*>(handle);if(!r)return -1;CUcontext old=nullptr;bool push=cuCtxGetCurrent(&old)!=CUDA_SUCCESS||old!=r->context;
 if(push&&cuCtxPushCurrent(r->context)!=CUDA_SUCCESS)return r->fail(824);
 r->barrier();if(r->phase&&!r->error)r->fail(825);int error=r->error;r->release();if(push)cuCtxPopCurrent(&old);delete r;return error;
}

extern "C" int robot_surface_runtime_component_trace(void* handle,unsigned long long* out,unsigned rounds){
 auto r=static_cast<Runtime*>(handle);if(!r||!out||!rounds||rounds>256)return -1;
 CUcontext old=nullptr;bool push=cuCtxGetCurrent(&old)!=CUDA_SUCCESS||old!=r->context;
 if(push&&cuCtxPushCurrent(r->context)!=CUDA_SUCCESS)return r->fail(824);
 int code=r->barrier();CUdeviceptr p=0;size_t bytes=0;
 if(!code&&(cuModuleGetGlobal(&p,&bytes,r->module,"component_trace")!=CUDA_SUCCESS||bytes!=256*24*8||cuMemcpyDtoH(out,p,rounds*24*8)!=CUDA_SUCCESS))code=r->fail(843);
 if(push)cuCtxPopCurrent(&old);return code;
}
