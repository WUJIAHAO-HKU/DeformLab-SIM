// Resident host bridge for the pinned finite-table row primitive.
// Caller owns native query/solve launches. This bridge owns only staging/cache.
#include <cuda.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
struct Row {unsigned node,tet,corner,source;};
struct Runtime {
 CUcontext context=nullptr;CUmodule module=nullptr;CUstream stream=nullptr;
 unsigned nodes=0,tets=0,boundary_count=0,capacity=0,frame=0,round=0,phase=0;
 bool begun=false,stream_bound=false;int error=0;
 double cx=0,cy=0,hx=0,hy=0,plane=0,epsilon=2e-7;
 CUdeviceptr header=0,cache=0,seen=0,active=0,boundary=0,plan=0,selected=0,zero=0;
 CUdeviceptr info=0,sid=0,constraints=0,lambda=0,rigid=0,count=0;
 CUdeviceptr input[9]={};unsigned current[4]={};
 unsigned long long prepared=0,published=0,added=0,resets=0,bytes=0;
 struct Allocation {CUdeviceptr ptr;size_t size;};std::vector<Allocation> allocations;std::vector<unsigned> fixed_topology;
 int fail(int code){if(!error)error=code;return error;}
 int check(){CUcontext c=nullptr;if(error)return error;if(cuCtxGetCurrent(&c)!=CUDA_SUCCESS||c!=context)return fail(801);return 0;}
 bool allocate(CUdeviceptr& p,size_t size){
  if(cuMemAlloc(&p,size+64)!=CUDA_SUCCESS)return false;allocations.push_back({p,size});bytes+=size+64;
  return cuMemsetD8(p,0,size)==CUDA_SUCCESS&&cuMemsetD8(p+size,0xa5,64)==CUDA_SUCCESS;
 }
 bool pointer(CUdeviceptr p,size_t size){
  CUcontext c=nullptr;CUdeviceptr base=0;size_t total=0;
  return p&&cuPointerGetAttribute(&c,CU_POINTER_ATTRIBUTE_CONTEXT,p)==CUDA_SUCCESS&&c==context&&
   cuMemGetAddressRange(&base,&total,p)==CUDA_SUCCESS&&p>=base&&size<=total&&p-base<=total-size;
 }
 int launch(const char* name,void** args,unsigned threads=1){
  CUfunction f=nullptr;if(cuModuleGetFunction(&f,module,name)!=CUDA_SUCCESS)return fail(802);
  auto c=cuLaunchKernel(f,std::max(1u,(threads+127)/128),1,1,128,1,1,0,stream,args,nullptr);
  return c==CUDA_SUCCESS?0:fail(10000+int(c));
 }
 int sync(){auto c=cuStreamSynchronize(stream);return c==CUDA_SUCCESS?0:fail(11000+int(c));}
 int guards(){
  if(sync())return error;unsigned char b[64];
  for(auto a:allocations){if(cuMemcpyDtoH(b,a.ptr+a.size,64)!=CUDA_SUCCESS)return fail(803);for(auto v:b)if(v!=0xa5)return fail(804);}return 0;
 }
 int header_check(unsigned wanted_phase){
  unsigned h[8];if(sync()||cuMemcpyDtoH(h,header,32)!=CUDA_SUCCESS)return fail(805);
  if(h[6])return fail(20000+int(h[6]));
  if(h[0]!=0x53555246||h[1]!=frame||h[2]!=round||h[3]!=wanted_phase||h[4]!=nodes||h[5]!=boundary_count)return fail(806);
  return 0;
 }
 int copy(CUdeviceptr dst,CUdeviceptr src,size_t size){
  if(!size)return 0;auto c=cuMemcpyDtoDAsync(dst,src,size,stream);return c==CUDA_SUCCESS?0:fail(12000+int(c));
 }
 void release(){for(auto a:allocations)cuMemFree(a.ptr);allocations.clear();if(module)cuModuleUnload(module);module=nullptr;}
};
extern "C" void* robot_surface_runtime_create(const char* ptx,unsigned nodes,unsigned tets,const Row* boundary,unsigned count,double cx,double cy,double hx,double hy,double plane){
 if(!ptx||!*ptx||!nodes||nodes>65536||!tets||tets>1048576||!count||count>nodes||!boundary||!std::isfinite(cx)||!std::isfinite(cy)||!std::isfinite(hx)||!std::isfinite(hy)||!std::isfinite(plane)||hx<=0||hy<=0)return nullptr;
 for(unsigned i=0;i<count;++i)if(boundary[i].node>=nodes||boundary[i].tet>=tets||boundary[i].corner>=4||(i&&boundary[i].node<=boundary[i-1].node))return nullptr;
 auto r=new Runtime;r->nodes=nodes;r->tets=tets;r->boundary_count=count;r->capacity=16384+count;r->cx=cx;r->cy=cy;r->hx=hx;r->hy=hy;r->plane=plane;
 if(cuCtxGetCurrent(&r->context)!=CUDA_SUCCESS||!r->context||cuModuleLoad(&r->module,ptx)!=CUDA_SUCCESS){r->release();delete r;return nullptr;}
 unsigned cap=r->capacity;
 bool ok=r->allocate(r->header,32)&&r->allocate(r->cache,nodes*4)&&r->allocate(r->seen,nodes*4)&&r->allocate(r->active,nodes*4)&&
  r->allocate(r->boundary,count*16)&&r->allocate(r->plan,16)&&r->allocate(r->selected,count*16)&&r->allocate(r->zero,cap*32)&&
  r->allocate(r->info,cap*16)&&r->allocate(r->sid,cap*4)&&r->allocate(r->constraints,((cap+31)/32)*3200)&&
  r->allocate(r->lambda,cap*4)&&r->allocate(r->rigid,cap*32)&&r->allocate(r->count,4);
 if(!ok||cuMemcpyHtoD(r->boundary,boundary,count*16)!=CUDA_SUCCESS){r->release();delete r;return nullptr;}return r;
}
extern "C" int robot_surface_runtime_begin(void* handle,unsigned frame){
 auto r=static_cast<Runtime*>(handle);if(!r)return -1;if(r->check())return r->error;
 if(r->phase||(r->begun&&frame<=r->frame))return r->fail(810);
 r->frame=frame;r->round=0;r->begun=true;
 void* args[]={&r->header,&r->cache,&r->seen,&r->active,&r->nodes,&r->boundary_count,&r->frame};
 if(r->launch("robot_surface_history_reset",args)||r->header_check(1))return r->error;++r->resets;return 0;
}
// Inputs: info, solver ids, constraints, lambda, native count, tets, position,
// delta_position, reserved(0). Reaction pointer is bound after query, because
// the native query ABI does not supply that pointer.
extern "C" int robot_surface_runtime_prepare(void* handle,const CUdeviceptr* input,unsigned frame,unsigned round,CUstream stream){
 auto r=static_cast<Runtime*>(handle);if(!r||!input)return -1;if(r->check())return r->error;
 if(!r->begun||r->phase||frame!=r->frame||round!=r->round||input[8])return r->fail(811);
 CUstreamCaptureStatus capture;if(cuStreamIsCapturing(stream,&capture)!=CUDA_SUCCESS||capture!=CU_STREAM_CAPTURE_STATUS_NONE)return r->fail(812);
 if(r->stream_bound&&stream!=r->stream)return r->fail(813);r->stream=stream;r->stream_bound=true;
 if(r->sync()||!r->pointer(input[4],4))return r->fail(814);unsigned n=0;if(cuMemcpyDtoH(&n,input[4],4)!=CUDA_SUCCESS||n>=16384)return r->fail(815);
 size_t sizes[]={size_t(n)*16,size_t(n)*4,size_t((n+31)/32)*3200,size_t(n)*4,4,size_t(r->tets)*16,size_t(r->nodes)*16,size_t(r->nodes)*16};
 for(unsigned i=0;i<8;++i)if(!r->pointer(input[i],sizes[i]))return r->fail(816);
 std::vector<unsigned> topology(size_t(r->tets)*4);
 if(cuMemcpyDtoH(topology.data(),input[5],topology.size()*4)!=CUDA_SUCCESS)return r->fail(826);
 if(r->fixed_topology.empty())r->fixed_topology=topology;
 else if(topology!=r->fixed_topology)return r->fail(827);
 std::vector<unsigned> ids(size_t(n)*4);
 if(n&&cuMemcpyDtoH(ids.data(),input[0],size_t(n)*16)!=CUDA_SUCCESS)return r->fail(828);
 for(unsigned i=0;i<n;++i)if((ids[4*i+2]>>20)!=0||(ids[4*i+2]&0xfffffu)>=r->tets)return r->fail(829);
 std::memcpy(r->input,input,sizeof(r->input));
 void* plan[]={&r->input[4],&r->input[0],&r->input[1],&r->input[2],&r->input[5],&r->input[6],&r->input[7],&r->boundary,&r->boundary_count,&r->nodes,&r->tets,&r->cx,&r->cy,&r->hx,&r->hy,&r->plane,&r->epsilon,&r->capacity,&r->plan,&r->selected,&r->cache,&r->seen,&r->header,&r->frame,&r->round};
 if(r->launch("robot_surface_plan",plan)||r->sync()||cuMemcpyDtoH(r->current,r->plan,16)!=CUDA_SUCCESS)return r->fail(817);
 if(r->current[0])return r->fail(30000+int(r->current[0]));
 void* copy[]={&r->plan,&r->input[0],&r->input[1],&r->input[2],&r->input[3],&r->zero,&r->info,&r->sid,&r->constraints,&r->lambda,&r->rigid,&r->count};
 unsigned padding=0;void* append[]={&r->plan,&r->selected,&r->input[0],&r->input[2],&r->input[6],&r->input[7],&r->plane,&padding,&r->info,&r->constraints};
 void* restore[]={&r->plan,&r->selected,&r->header,&r->cache,&r->active,&r->lambda,&r->info};
 if(r->launch("robot_surface_copy",copy,((r->current[2]+31)/32)*800)||r->launch("robot_surface_append",append,r->boundary_count)||r->launch("robot_surface_history_restore",restore,r->boundary_count)||r->header_check(2)||r->guards())return r->error;
 r->phase=1;++r->prepared;r->added+=r->current[2]-r->current[1];return 0;
}
extern "C" int robot_surface_runtime_outputs(void* handle,CUdeviceptr* pointers,unsigned* counts){
 auto r=static_cast<Runtime*>(handle);if(!r||!pointers||!counts)return -1;if(r->check())return r->error;
 if(!r->phase)return r->fail(818);CUdeviceptr p[]={r->info,r->sid,r->constraints,r->lambda,r->rigid,r->count};std::memcpy(pointers,p,sizeof(p));std::memcpy(counts,r->current,16);return 0;
}
extern "C" int robot_surface_runtime_after_query(void* handle){
 auto r=static_cast<Runtime*>(handle);if(!r)return -1;if(r->check())return r->error;if(r->phase!=1)return r->fail(819);
 if(r->guards()||r->copy(r->input[0],r->info,size_t(r->current[1])*16))return r->error;r->phase=2;return 0;
}
extern "C" int robot_surface_runtime_bind_reaction(void* handle,CUdeviceptr original){
 auto r=static_cast<Runtime*>(handle);if(!r)return -1;if(r->check())return r->error;if(r->phase!=2||!r->pointer(original,size_t(r->current[1])*32))return r->fail(820);
 r->input[8]=original;size_t bytes=size_t(r->current[1])*16;
 if(r->copy(r->rigid,original,bytes)||r->copy(r->rigid+size_t(r->current[2])*16,original+bytes,bytes))return r->error;r->phase=3;return 0;
}
extern "C" int robot_surface_runtime_publish(void* handle){
 auto r=static_cast<Runtime*>(handle);if(!r)return -1;if(r->check())return r->error;if(r->phase!=3)return r->fail(821);
 // Bound synchronous check in the diagnostic bridge: the full native reaction
 // chain may omit added rows only when both static payload planes are zero.
 unsigned n=r->current[1],nc=r->current[2];std::vector<float> tail(size_t(nc-n)*4);
 if(r->sync())return r->error;
 for(unsigned plane=0;plane<2;++plane){
  if(!tail.empty()&&cuMemcpyDtoH(tail.data(),r->rigid+(size_t(plane)*nc+n)*16,tail.size()*4)!=CUDA_SUCCESS)return r->fail(822);
  for(float v:tail)if(v!=0.f)return r->fail(823);
 }
 void* store[]={&r->plan,&r->selected,&r->header,&r->lambda,&r->info,&r->cache,&r->seen,&r->active};
 if(r->launch("robot_surface_history_store",store))return r->error;++r->round;
 if(r->header_check(1)||r->guards())return r->error;
 if(r->copy(r->input[0],r->info,size_t(n)*16)||r->copy(r->input[3],r->lambda,size_t(n)*4)||r->copy(r->input[8],r->rigid,size_t(n)*16)||r->copy(r->input[8]+size_t(n)*16,r->rigid+size_t(nc)*16,size_t(n)*16)||r->sync())return r->error;
 r->phase=0;++r->published;return 0;
}
extern "C" int robot_surface_runtime_stats(void* handle,unsigned long long* values){
 auto r=static_cast<Runtime*>(handle);if(!r||!values)return -1;
 unsigned long long v[]={r->prepared,r->published,r->added,r->resets,r->bytes,r->phase,unsigned(r->error),r->frame,r->round};std::memcpy(values,v,sizeof(v));return 0;
}
extern "C" int robot_surface_runtime_stop(void* handle){
 auto r=static_cast<Runtime*>(handle);if(!r)return -1;
 CUcontext old=nullptr;bool push=cuCtxGetCurrent(&old)!=CUDA_SUCCESS||old!=r->context;
 if(push&&cuCtxPushCurrent(r->context)!=CUDA_SUCCESS)return r->fail(824);
 if(r->phase&&!r->error)r->fail(825);r->sync();int error=r->error;r->release();if(push)cuCtxPopCurrent(&old);delete r;return error;
}
