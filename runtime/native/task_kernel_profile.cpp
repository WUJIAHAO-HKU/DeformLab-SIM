// Read-only host launch census. Never replaces a kernel, parameters or stream.
// Graph/extended launches are forwarded but mark the census incomplete.
#include <cuda.h>
#undef cuGetProcAddress
#include <dlfcn.h>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <algorithm>
using Launch=CUresult(CUDAAPI*)(CUfunction,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,CUstream,void**,void**);
using Lookup=void*(*)(void*,const char*);
static Lookup lookup(){static auto p=reinterpret_cast<Lookup>(dlvsym(RTLD_NEXT,"dlsym","GLIBC_2.2.5"));return p;}
namespace {
std::atomic<bool> enabled{false};std::mutex mutex;FILE* output=nullptr;
struct Counts {unsigned long long attempted=0,succeeded=0;unsigned min_grid=~0u,max_grid=0,min_block=~0u,max_block=0;};
std::map<std::string,Counts> counters;unsigned errors=0,unsupported=0;
CUresult counted(const char* entry,CUfunction f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned shared,CUstream stream,void** params,void** extra){
 auto original=reinterpret_cast<Launch>(lookup()(RTLD_NEXT,entry));
 if(!enabled)return original(f,gx,gy,gz,bx,by,bz,shared,stream,params,extra);
 std::lock_guard<std::mutex> guard(mutex);
 if(!enabled)return original(f,gx,gy,gz,bx,by,bz,shared,stream,params,extra);
 const char* name=nullptr;CUcontext context=nullptr;
 bool known=cuFuncGetName(&name,f)==CUDA_SUCCESS&&name&&cuCtxGetCurrent(&context)==CUDA_SUCCESS;
 char context_name[32];std::snprintf(context_name,sizeof(context_name),"%p",reinterpret_cast<void*>(context));
 if(!known||std::strpbrk(name,"\t\r\n")){++errors;known=false;}
 auto result=original(f,gx,gy,gz,bx,by,bz,shared,stream,params,extra);
 if(known){auto& c=counters[std::string(context_name)+"\t"+name];++c.attempted;if(result==CUDA_SUCCESS)++c.succeeded;else ++errors;
  c.min_grid=std::min(c.min_grid,gx);c.max_grid=std::max(c.max_grid,gx);c.min_block=std::min(c.min_block,bx);c.max_block=std::max(c.max_block,bx);}
 return result;
}
void unseen(){if(enabled){std::lock_guard<std::mutex> guard(mutex);if(enabled)++unsupported;}}
}
extern "C" CUresult CUDAAPI cuLaunchKernel(CUfunction f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned sh,CUstream s,void** p,void** e){return counted("cuLaunchKernel",f,gx,gy,gz,bx,by,bz,sh,s,p,e);}
extern "C" CUresult CUDAAPI cuLaunchKernel_ptsz(CUfunction f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned sh,CUstream s,void** p,void** e){return counted("cuLaunchKernel_ptsz",f,gx,gy,gz,bx,by,bz,sh,s,p,e);}
extern "C" CUresult CUDAAPI cuGraphLaunch(CUgraphExec g,CUstream s){unseen();using Fn=CUresult(CUDAAPI*)(CUgraphExec,CUstream);return reinterpret_cast<Fn>(lookup()(RTLD_NEXT,"cuGraphLaunch"))(g,s);}
extern "C" CUresult CUDAAPI cuGraphLaunch_ptsz(CUgraphExec g,CUstream s){unseen();using Fn=CUresult(CUDAAPI*)(CUgraphExec,CUstream);return reinterpret_cast<Fn>(lookup()(RTLD_NEXT,"cuGraphLaunch_ptsz"))(g,s);}
extern "C" CUresult CUDAAPI cuLaunchKernelEx(const CUlaunchConfig* c,CUfunction f,void** p,void** e){unseen();using Fn=CUresult(CUDAAPI*)(const CUlaunchConfig*,CUfunction,void**,void**);return reinterpret_cast<Fn>(lookup()(RTLD_NEXT,"cuLaunchKernelEx"))(c,f,p,e);}
extern "C" CUresult CUDAAPI cuLaunchKernelEx_ptsz(const CUlaunchConfig* c,CUfunction f,void** p,void** e){unseen();using Fn=CUresult(CUDAAPI*)(const CUlaunchConfig*,CUfunction,void**,void**);return reinterpret_cast<Fn>(lookup()(RTLD_NEXT,"cuLaunchKernelEx_ptsz"))(c,f,p,e);}
extern "C" CUresult CUDAAPI cuGetProcAddress(const char*,void**,int,cuuint64_t);
static void* substitute(const char* name,void* original){
 if(!original)return original;
#define SUBSTITUTE(symbol) if(!std::strcmp(name,#symbol))return reinterpret_cast<void*>(&symbol);
 SUBSTITUTE(cuLaunchKernel) SUBSTITUTE(cuLaunchKernel_ptsz) SUBSTITUTE(cuGraphLaunch) SUBSTITUTE(cuGraphLaunch_ptsz)
 SUBSTITUTE(cuLaunchKernelEx) SUBSTITUTE(cuLaunchKernelEx_ptsz) SUBSTITUTE(cuGetProcAddress) SUBSTITUTE(cuGetProcAddress_v2)
#undef SUBSTITUTE
 return original;
}
static void* proc_substitute(const char* name,void* original,cuuint64_t flags){
 if(!original)return original;
 if(flags&CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM){
  if(!std::strcmp(name,"cuLaunchKernel"))return reinterpret_cast<void*>(&cuLaunchKernel_ptsz);
  if(!std::strcmp(name,"cuLaunchKernelEx"))return reinterpret_cast<void*>(&cuLaunchKernelEx_ptsz);
  if(!std::strcmp(name,"cuGraphLaunch"))return reinterpret_cast<void*>(&cuGraphLaunch_ptsz);
 }
 return substitute(name,original);
}
extern "C" void* dlsym(void* h,const char* n) noexcept{return substitute(n,lookup()(h,n));}
extern "C" CUresult CUDAAPI cuGetProcAddress(const char* name,void** out,int version,cuuint64_t flags){
 using Fn=CUresult(CUDAAPI*)(const char*,void**,int,cuuint64_t);auto fn=reinterpret_cast<Fn>(lookup()(RTLD_NEXT,"cuGetProcAddress"));auto c=fn(name,out,version,flags);
 if(c==CUDA_SUCCESS&&out)*out=proc_substitute(name,*out,flags);return c;
}
extern "C" CUresult CUDAAPI cuGetProcAddress_v2(const char* name,void** out,int version,cuuint64_t flags,CUdriverProcAddressQueryResult* status){
 using Fn=CUresult(CUDAAPI*)(const char*,void**,int,cuuint64_t,CUdriverProcAddressQueryResult*);auto fn=reinterpret_cast<Fn>(lookup()(RTLD_NEXT,"cuGetProcAddress_v2"));auto c=fn(name,out,version,flags,status);
 if(c==CUDA_SUCCESS&&out)*out=proc_substitute(name,*out,flags);return c;
}
extern "C" int task_kernel_profile_start(const char* path){
 std::lock_guard<std::mutex> guard(mutex);if(enabled||output||!path||!*path)return -1;
 output=std::fopen(path,"wx");if(!output)return -2;
 counters.clear();errors=unsupported=0;enabled=true;return 0;
}
extern "C" int task_kernel_profile_stop(){
 std::lock_guard<std::mutex> guard(mutex);if(!enabled||!output)return -1;enabled=false;
 if(std::fprintf(output,"context\tkernel\tattempted\tsucceeded\tmin_grid_x\tmax_grid_x\tmin_block_x\tmax_block_x\n")<0)++errors;
 for(const auto& item:counters){const auto& c=item.second;if(std::fprintf(output,"%s\t%llu\t%llu\t%u\t%u\t%u\t%u\n",item.first.c_str(),c.attempted,c.succeeded,c.min_grid,c.max_grid,c.min_block,c.max_block)<0)++errors;}
 if(std::fprintf(output,"# errors=%u unsupported_launches=%u\n",errors,unsupported)<0)++errors;
 if(std::fclose(output))++errors;output=nullptr;return int(errors+unsupported);
}
