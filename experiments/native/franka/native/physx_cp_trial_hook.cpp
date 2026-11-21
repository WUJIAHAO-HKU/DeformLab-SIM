// Experimental Linux driver-API interposer. No CUPTI callback launches.
// Not a supported PhysX extension or a portable/production solver patch.
#include <cuda.h>
#include <algorithm>
#include <vector_types.h>
#undef cuGetProcAddress
#include <dlfcn.h>
#include <atomic>
#include <cstring>
#include <mutex>
#include <cstdio>
#include <vector>
#include <cmath>
#include <string>
static void audit_cp_partition(CUstream,void**,bool);
using Launch=CUresult(CUDAAPI*)(CUfunction,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,CUstream,void**,void**);
using Lookup=void*(*)(void*,const char*);
static Lookup real_lookup(){static auto p=reinterpret_cast<Lookup>(dlvsym(RTLD_NEXT,"dlsym","GLIBC_2.2.5"));return p;}
static Launch real_launch(){static auto p=reinterpret_cast<Launch>(real_lookup()(RTLD_NEXT,"cuLaunchKernel"));return p;}
static Launch real_launch_ptsz(){static auto p=reinterpret_cast<Launch>(real_lookup()(RTLD_NEXT,"cuLaunchKernel_ptsz"));return p;}
namespace {
std::atomic<bool> enabled{false};std::atomic<unsigned> errors{0};
std::atomic<unsigned long long> active_calls{0},context_calls{0},soft_calls{0};
CUcontext context=nullptr;CUmodule module=nullptr;
CUdeviceptr positions=0,velocities=0,sb=0,saved=0,scratch=0;
CUfunction splitfn=nullptr,preparefn=nullptr,mergefn=nullptr,cachefn=nullptr,dampingfn=nullptr;
CUdeviceptr tet_cache=0,rejected=0;unsigned tets=0;unsigned long long damping_calls=0;
CUdeviceptr residual=0,residual_cache=0;CUfunction capturefn=nullptr,publishfn=nullptr,residualmergefn=nullptr;
CUdeviceptr precise_p=0,precise_v=0;
unsigned long long residual_counts[3]={};
unsigned long long ordered_contact_calls=0;
size_t nodes=0;int mode=0;bool coupled_damping=false;unsigned long long counts[3]={0,0,0};
CUstream solver_stream=nullptr;bool stream_seen=false;
std::mutex lock;
template<class T>T field(const unsigned char* p,size_t off){T x;std::memcpy(&x,p+off,sizeof(x));return x;}
CUresult inject(CUfunction f,CUstream stream,void** params){return real_launch()(f,unsigned((nodes+127)/128),1,1,128,1,1,0,stream,params,nullptr);}
}
#include "physx_contact_policy.inc"
#include "physx_plane_coverage.inc"
#include "physx_rigid_reduction.inc"
#include "physx_friction_trial.inc"
#include "physx_normal_contact_trial.inc"
#include "physx_robot_surface_stream_adapter.inc"
namespace {bool block_before(void**,CUstream);bool block_after(CUstream);bool block_after_capture(CUstream);bool block_after_apply(CUstream);bool block_after_reduce(CUstream);void block_stop();}
namespace {
CUresult wrapped_impl(Launch original,CUfunction f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned shared,CUstream stream,void** params,void** extra){
    if(!enabled.load())return original(f,gx,gy,gz,bx,by,bz,shared,stream,params,extra);
    ++active_calls;
    CUcontext current;CUresult c=cuCtxGetCurrent(&current);if(c!=CUDA_SUCCESS)return c;
    if(current!=context)return original(f,gx,gy,gz,bx,by,bz,shared,stream,params,extra);
    ++context_calls;
    const char* name=nullptr;c=cuFuncGetName(&name,f);if(c!=CUDA_SUCCESS)return c;
    if(std::strstr(name,"sb_") && soft_calls.fetch_add(1)<4)std::fprintf(stderr,"CP trial kernel name: %s\n",name);
    if(rs_runtime && (!std::strcmp(name,"sb_queryRigidSoftContactReferenceCountLaunchTGS") || !std::strcmp(name,"sb_solveRigidSoftCollisionLaunchTGS")))
        return rs_dispatch(original,f,!std::strcmp(name,"sb_queryRigidSoftContactReferenceCountLaunchTGS"),gx,gy,gz,bx,by,bz,shared,stream,params,extra);
    if(normal_policy && !std::strcmp(name,"sb_solveRigidSoftCollisionLaunchTGS")){
        if(!params || extra || !stream){++errors;return CUDA_ERROR_NOT_SUPPORTED;}
        auto result=original(normal_function,gx,gy,gz,bx,by,bz,shared,stream,params,nullptr);
        if(result!=CUDA_SUCCESS)++errors;else ++normal_calls;return result;
    }
    if(rr_policy && (!std::strcmp(name,"sb_solveRigidSoftCollisionLaunchTGS")||!std::strcmp(name,"accumulateDeltaVRigidSecondLaunch"))){
        std::lock_guard<std::mutex> guard(rr_lock);
        if(!params||extra||!stream){++errors;return CUDA_ERROR_NOT_SUPPORTED;}
        bool solving=!std::strcmp(name,"sb_solveRigidSoftCollisionLaunchTGS"),matched=solving||rr_matches(params);
        if(solving&&!block_before(params,stream)){++errors;return CUDA_ERROR_NOT_SUPPORTED;}
        auto result=original(solving&&friction_policy?friction_function:f,gx,gy,gz,bx,by,bz,shared,stream,params,nullptr);
        if(result==CUDA_SUCCESS&&solving&&!block_after(stream)){++errors;return CUDA_ERROR_NOT_SUPPORTED;}
        if(result==CUDA_SUCCESS&&solving&&friction_policy)++friction_calls;
        if(result==CUDA_SUCCESS&&matched)result=solving?rr_after_solve(stream,params):rr_after_reduce(stream,params);
        if(result==CUDA_SUCCESS&&solving&&!block_after_capture(stream)){++errors;return CUDA_ERROR_NOT_SUPPORTED;}
        if(result==CUDA_SUCCESS&&!solving&&matched&&!block_after_reduce(stream)){++errors;return CUDA_ERROR_NOT_SUPPORTED;}
        if(result!=CUDA_SUCCESS)++errors;return result;
    }
    if(pc_policy && !std::strcmp(name,"sb_rigidContactPrepareLaunch")){
        if(!params || extra || !stream){++errors;return CUDA_ERROR_NOT_SUPPORTED;}
        if(!pc_prepare(params)){
            ++errors;
            // The Python frame boundary rejects this trial immediately. Let
            // valid original inputs prepare constraints for orderly shutdown;
            // skipping prepare would leave downstream GPU buffers undefined.
            if(pc_inputs_usable)return original(f,gx,gy,gz,bx,by,bz,shared,stream,params,nullptr);
            return CUDA_ERROR_NOT_SUPPORTED;
        }
    }
    if(cp_policy && !std::strcmp(name,"sb_rigidContactPrepareLaunch")){
        if(extra || !stream || !cp_prepare(params)){++errors;return CUDA_ERROR_NOT_SUPPORTED;}
    }
    if((cp_policy==2||cp_policy==4) && !std::strcmp(name,"sb_queryRigidSoftContactReferenceCountLaunchTGS")){
        if(!params||extra||!stream){++errors;return CUDA_ERROR_NOT_SUPPORTED;}
        auto result=original(f,gx,gy,gz,bx,by,bz,shared,stream,params,nullptr);
        if(result==CUDA_SUCCESS){void* p[]={params[0],params[1],params[2],params[3],&rejected};result=real_launch()(cp_countfn,128,1,1,128,1,1,0,stream,p,nullptr);}
        if(result!=CUDA_SUCCESS)++errors;else ++cp_queries;return result;
    }
    bool pre=!std::strcmp(name,"sb_gm_preIntegrateLaunch");
    bool partition=!std::strcmp(name,"sb_gm_cp_solveTetrahedronsPartitionLaunch");
    bool average=!std::strcmp(name,"sb_gm_cp_averageVertsLaunch");
    bool finalize=!std::strcmp(name,"sb_gm_finalizeVelocitiesLaunch");
    bool step=mode>=8 && !std::strcmp(name,"sb_gm_stepSoftbodyLaunch");
    bool contact=mode>=8 && !std::strcmp(name,"sb_gm_applyExternalDeltasLaunch");
    // Diagnostic only: fixed one-warp grid-stride contact order. Same contact
    // equations and reduction path; no extra solve or time advancement.
    if(mode==11 && !std::strcmp(name,"sb_solveRigidSoftCollisionLaunchTGS")){
        if(!params || extra || !stream || gy!=1 || gz!=1 || by!=1 || bz!=1){++errors;return CUDA_ERROR_NOT_SUPPORTED;}
        auto result=original(f,1,1,1,32,1,1,shared,stream,params,nullptr);
        if(result!=CUDA_SUCCESS)++errors;else ++ordered_contact_calls;return result;
    }
    if(!pre&&!partition&&!average&&!finalize&&!step&&!contact)return original(f,gx,gy,gz,bx,by,bz,shared,stream,params,extra);
    std::lock_guard<std::mutex> guard(lock);
    auto fail=[&](){if(errors.fetch_add(1)==0)std::fprintf(stderr,"CP trial rejected launch: %s grid=%u,%u,%u stream=%p solver=%p\n",name,gx,gy,gz,static_cast<void*>(stream),static_cast<void*>(solver_stream));return CUDA_ERROR_NOT_SUPPORTED;};
    if(!params || extra || gy!=1 || gz!=1 || !stream)return fail();
    CUstreamCaptureStatus capture;if(cuStreamIsCapturing(stream,&capture)!=CUDA_SUCCESS || capture!=CU_STREAM_CAPTURE_STATUS_NONE)return fail();
    if(partition || average){if(stream_seen && stream!=solver_stream)return fail();solver_stream=stream;stream_seen=true;}
    if(pre){
        unsigned id=0;auto ids=*static_cast<const CUdeviceptr*>(params[1]);auto base=*static_cast<const CUdeviceptr*>(params[0]);
        if(cuStreamSynchronize(stream)!=CUDA_SUCCESS || cuMemcpyDtoH(&id,ids,4)!=CUDA_SUCCESS)return fail();
        auto found=base+size_t(id)*432;if(sb && found!=sb)return fail();sb=found;
        unsigned char data[432];if(cuMemcpyDtoH(data,sb,sizeof(data))!=CUDA_SUCCESS)return fail();
        if(field<CUdeviceptr>(data,88)!=velocities || field<CUdeviceptr>(data,96)!=positions || field<unsigned>(data,372)!=nodes || field<unsigned>(data,412)!=1 || field<unsigned>(data,380)>8)return fail();
    }
    if(!sb)return fail();
    int residual_op=pre?0:step?1:2;
    if(mode>=8 && pre){
        unsigned char data[432];if(cuMemcpyDtoH(data,sb,432)!=CUDA_SUCCESS)return fail();
        if(!*static_cast<unsigned char*>(params[4]) || (data[393]&4))return fail();
    }
    if(mode>=8 && (pre || step || contact)){
        void* p[]={&sb,&residual_cache,&residual_op,&rejected};c=inject(capturefn,stream,p);if(c!=CUDA_SUCCESS){++errors;return c;}
    }
    if(finalize && *static_cast<const unsigned char*>(params[5]))return fail();
    if(mode==3){if(average)++counts[1];c=original(f,gx,gy,gz,bx,by,bz,shared,stream,params,extra);if(c!=CUDA_SUCCESS)++errors;else if(average)++counts[2];return c;}
    if(partition && (mode==2 || mode==4 || mode==5 || mode==6 || mode==7 || mode>=8) && *static_cast<const unsigned*>(params[4])==0){void* p[]={&sb,&saved};c=inject(splitfn,stream,p);if(c!=CUDA_SUCCESS){++errors;return c;}++counts[0];}
    if(average){int split=(mode==2 || mode==4 || mode==5 || mode==6 || mode==7 || mode>=8);void* p[]={&sb,&saved,&scratch,&split};c=inject(preparefn,stream,p);if(c!=CUDA_SUCCESS){++errors;return c;}++counts[1];}
    if(partition && (mode==4 || mode==5 || mode==6 || mode==7 || mode>=8)){
        if(!*static_cast<unsigned char*>(params[5]) || *static_cast<unsigned*>(params[2])!=1)return fail();
        if(!tet_cache){
            unsigned char data[432];if(cuStreamSynchronize(stream)!=CUDA_SUCCESS || cuMemcpyDtoH(data,sb,432)!=CUDA_SUCCESS)return fail();
            tets=field<unsigned>(data,376);if(!tets || tets>1000000)return fail();
            if(cuMemAlloc(&tet_cache,size_t(tets)*(mode>=9?256:128))!=CUDA_SUCCESS)return fail();
            if(mode>=9){if(cuMemAlloc(&precise_p,size_t(tets)*8*32)!=CUDA_SUCCESS || cuMemAlloc(&precise_v,size_t(tets)*8*32)!=CUDA_SUCCESS)return fail();}
        }
        unsigned part=*static_cast<unsigned*>(params[4]);void* p[]={&sb,&tet_cache,&part,&residual,&saved,&precise_p,&precise_v};
        c=real_launch()(cachefn,(tets+127)/128,1,1,128,1,1,0,stream,p,nullptr);if(c!=CUDA_SUCCESS){++errors;return c;}
    }
    if(partition)audit_cp_partition(stream,params,false);
    c=original(f,gx,gy,gz,bx,by,bz,shared,stream,params,extra);
    if(c!=CUDA_SUCCESS){++errors;return c;}
    if(partition && (mode==4 || mode==5 || mode==6 || mode==7 || mode>=8)){
        unsigned part=*static_cast<unsigned*>(params[4]);float invdt=*static_cast<float*>(params[3]);auto mat=*static_cast<CUdeviceptr*>(params[7]);int sequential=mode==7?3:(mode==6 || mode>=8)?2:(mode==4?1:0);
        if(coupled_damping)sequential=4;
        void* p[]={&sb,&tet_cache,&part,&invdt,&mat,&sequential,&rejected,&precise_p,&precise_v};
        c=real_launch()(dampingfn,(tets+127)/128,1,1,128,1,1,0,stream,p,nullptr);if(c!=CUDA_SUCCESS){++errors;return c;}++damping_calls;
    }
    if(partition)audit_cp_partition(stream,params,true);
    if(average){float invdt=*static_cast<const float*>(params[2]);void* p[]={&sb,&scratch,&invdt,&residual,&precise_v};c=inject(mode>=8?residualmergefn:mergefn,stream,p);if(c!=CUDA_SUCCESS)++errors;else ++counts[2];}
    if(mode>=8 && (pre || step || contact)){
        float dt=(step || (mode>=9 && contact))?*static_cast<float*>(params[2]):0.f;
        void* p[]={&sb,&residual_cache,&residual,&residual_op,&dt,&rejected};
        c=inject(publishfn,stream,p);if(c!=CUDA_SUCCESS)++errors;else ++residual_counts[residual_op];
        if(c==CUDA_SUCCESS&&contact&&!block_after_apply(stream)){++errors;return CUDA_ERROR_NOT_SUPPORTED;}
    }
    return c;
}
}
namespace {
CUresult wrapped(Launch original,CUfunction f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned shared,CUstream stream,void** params,void** extra){
 const char* name=nullptr;CUcontext current=nullptr;
 bool monitored=pc_policy==4&&enabled&&cuCtxGetCurrent(&current)==CUDA_SUCCESS&&current==context&&cuFuncGetName(&name,f)==CUDA_SUCCESS;
 if(monitored){if(!pc_guard_bind(name,params))++errors;if(pc_guard_selected(name)&&!pc_guard_check(name,"before"))++errors;}
 auto result=wrapped_impl(original,f,gx,gy,gz,bx,by,bz,shared,stream,params,extra);
 // Preserve native launch execution for orderly same-frame rejection.
 if(monitored&&result==CUDA_SUCCESS&&pc_guard_selected(name)&&!pc_guard_check(name,"after"))++errors;
 return result;
}
}
#include "physx_contact_observation.inc"
#include "physx_articulation_work.inc"
#include "physx_contact_block.inc"
#include "physx_angular_transport.inc"
namespace {
// Opt-in readback around ordinary host launch calls, separate from CUPTI.
// Fixed physical masses are used even while CP metadata carries split masses.
FILE* velocity_residual_states=nullptr;
FILE* residual_states=nullptr;FILE* auxiliary_states=nullptr;
FILE* trace_file=nullptr;FILE* angular_states=nullptr;std::atomic<int> trace_step{-1};unsigned long long trace_launch=0;
std::mutex trace_lock;
std::vector<double3> trace_residual;std::vector<float4> trace_auxiliary;
std::vector<double> trace_mass;std::vector<float4> trace_p,trace_v;
std::string audit_prefix;bool audit_done[8]={};int audit_sample_step=4800;
float material_youngs=NAN,material_poisson=NAN,material_damping=NAN;unsigned material_model=~0u;
bool trace_snapshot(CUfunction f,CUstream stream,void** params,const char* phase,unsigned long long ordinal){
    if(!trace_file || trace_step<0 || !enabled.load())return false;
    CUcontext current=nullptr;if(cuCtxGetCurrent(&current)!=CUDA_SUCCESS || current!=context)return false;
    const char* name=nullptr;
    if(cuFuncGetName(&name,f)!=CUDA_SUCCESS || cuStreamSynchronize(stream)!=CUDA_SUCCESS ||
       cuMemcpyDtoH(trace_p.data(),positions,nodes*16)!=CUDA_SUCCESS ||
       cuMemcpyDtoH(trace_v.data(),velocities,nodes*16)!=CUDA_SUCCESS){++errors;return false;}
    double momentum[3]={},angular[3]={},center[3]={},mass=0,ke=0,min_ratio=1e100,max_ratio=0,mass_mismatch=0;
    for(size_t i=0;i<nodes;++i){
        double m=trace_mass[i];mass+=m;const float* p=&trace_p[i].x;const float* v=&trace_v[i].x;
        for(int j=0;j<3;++j){momentum[j]+=m*v[j];center[j]+=m*p[j];ke+=.5*m*v[j]*v[j];}
        for(int j=0;j<3;++j)angular[j]+=m*(double(p[(j+1)%3])*v[(j+2)%3]-double(p[(j+2)%3])*v[(j+1)%3]);
        double ratio=m*trace_p[i].w;min_ratio=std::fmin(min_ratio,ratio);max_ratio=std::fmax(max_ratio,ratio);
        mass_mismatch=std::fmax(mass_mismatch,std::fabs(double(trace_p[i].w)-trace_v[i].w));
    }
    float invdt=NAN;int part=-1;
    if(params && !std::strcmp(name,"sb_gm_cp_solveTetrahedronsPartitionLaunch")){invdt=*static_cast<float*>(params[3]);part=*static_cast<unsigned*>(params[4]);}
    if(params && (!std::strcmp(name,"sb_gm_cp_averageVertsLaunch") || !std::strcmp(name,"sb_gm_applyExternalDeltasLaunch")))invdt=*static_cast<float*>(params[2]);
    if(std::fprintf(trace_file,"%d\t%llu\t%s\t%s\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.9g\t%d\t%.9g\t%.9g\t%.9g\t%u\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g\n",trace_step.load(),ordinal,phase,name,
        momentum[0],momentum[1],momentum[2],center[0]/mass,center[1]/mass,center[2]/mass,ke,invdt,part,material_youngs,material_poisson,material_damping,material_model,min_ratio,max_ratio,mass_mismatch,angular[0],angular[1],angular[2])<0)++errors;
    if(angular_states){
        if(std::fwrite(trace_p.data(),16,nodes,angular_states)!=nodes || std::fwrite(trace_v.data(),16,nodes,angular_states)!=nodes)++errors;
    }
    if(residual_states){
        // The first before-preIntegrate snapshot precedes sb discovery.
        CUdeviceptr body=sb;
        if(!body && params && !std::strcmp(name,"sb_gm_preIntegrateLaunch")){
            unsigned id;auto ids=*static_cast<CUdeviceptr*>(params[1]);
            if(cuMemcpyDtoH(&id,ids,4)!=CUDA_SUCCESS){++errors;return false;}
            body=*static_cast<CUdeviceptr*>(params[0])+size_t(id)*432;
        }
        unsigned char metadata[432];
        if(cuMemcpyDtoH(trace_residual.data(),residual,nodes*(mode>=9?72:48))!=CUDA_SUCCESS){++errors;return false;}
        if(body){
            if(cuMemcpyDtoH(metadata,body,432)!=CUDA_SUCCESS ||
               cuMemcpyDtoH(trace_auxiliary.data(),field<CUdeviceptr>(metadata,128),nodes*16)!=CUDA_SUCCESS ||
               cuMemcpyDtoH(trace_auxiliary.data()+nodes,field<CUdeviceptr>(metadata,272),nodes*16)!=CUDA_SUCCESS){++errors;return false;}
        }else{
            // Other native launches may precede the first soft-body launch.
            // The sidecar is known (zero); native auxiliary buffers are not yet
            // discovered. Keep row alignment and mark only those fields absent.
            for(auto& a:trace_auxiliary)a={NAN,NAN,NAN,NAN};
        }
        if(velocity_residual_states && std::fwrite(trace_residual.data()+2*nodes,24,nodes,velocity_residual_states)!=nodes)++errors;
        if(std::fwrite(trace_residual.data(),24,2*nodes,residual_states)!=2*nodes ||
           std::fwrite(trace_auxiliary.data(),16,2*nodes,auxiliary_states)!=2*nodes)++errors;
    }
    return true;
}
CUresult angular_traced_launch(Launch original,CUfunction f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned shared,CUstream stream,void** params,void** extra){
    if(trace_step.load()<0)return wrapped(original,f,gx,gy,gz,bx,by,bz,shared,stream,params,extra);
    std::lock_guard<std::mutex> guard(trace_lock);
    auto ordinal=trace_launch++;bool traced=trace_snapshot(f,stream,params,"before",ordinal);
    auto result=wrapped(original,f,gx,gy,gz,bx,by,bz,shared,stream,params,extra);
    if(traced)trace_snapshot(f,stream,params,"after",ordinal);
    return result;
}
CUresult traced_launch(Launch original,CUfunction f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned shared,CUstream stream,void** params,void** extra){
 if(ct_step.load()<0)return angular_traced_launch(original,f,gx,gy,gz,bx,by,bz,shared,stream,params,extra);
 std::lock_guard<std::mutex> guard(ct_lock);const char* name=nullptr;
 if(cuFuncGetName(&name,f)!=CUDA_SUCCESS){++errors;return CUDA_ERROR_NOT_SUPPORTED;}
 CUcontext current=nullptr;bool transport=at_active(name)&&cuCtxGetCurrent(&current)==CUDA_SUCCESS&&current==context;
 auto seq=ct_seq++;if(transport&&!at_snapshot(name,params,"before",seq)){++errors;return CUDA_ERROR_NOT_SUPPORTED;}
 bool artwork=aw_selected(name);
 if(artwork&&!aw_snapshot(name,params,"before",seq)){++errors;return CUDA_ERROR_NOT_SUPPORTED;}
 bool captured=ct_selected(name) && ct_snapshot(name,params,true,seq);
 auto result=angular_traced_launch(original,f,gx,gy,gz,bx,by,bz,shared,stream,params,extra);
 if(transport&&result==CUDA_SUCCESS&&(!at_after(name,params,seq)||!at_snapshot(name,params,"after",seq))){++errors;return CUDA_ERROR_NOT_SUPPORTED;}
 if(artwork&&result==CUDA_SUCCESS&&!aw_snapshot(name,params,"after",seq)){++errors;return CUDA_ERROR_NOT_SUPPORTED;}
 if(captured)ct_snapshot(name,params,false,seq);return result;
}
}
extern "C" CUresult CUDAAPI cuLaunchKernel(CUfunction f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned shared,CUstream stream,void** params,void** extra){return traced_launch(real_launch(),f,gx,gy,gz,bx,by,bz,shared,stream,params,extra);}
extern "C" CUresult CUDAAPI cuLaunchKernel_ptsz(CUfunction f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned shared,CUstream stream,void** params,void** extra){return traced_launch(real_launch_ptsz(),f,gx,gy,gz,bx,by,bz,shared,stream,params,extra);}
extern "C" CUresult CUDAAPI cuGetProcAddress(const char*,void**,int,cuuint64_t);
static void* substitute(const char* name,void* original){
    if(!original)return original;
    if(!std::strcmp(name,"cuLaunchKernel"))return reinterpret_cast<void*>(&cuLaunchKernel);
    if(!std::strcmp(name,"cuLaunchKernel_ptsz"))return reinterpret_cast<void*>(&cuLaunchKernel_ptsz);
    if(!std::strcmp(name,"cuGetProcAddress"))return reinterpret_cast<void*>(&cuGetProcAddress);
    if(!std::strcmp(name,"cuGetProcAddress_v2"))return reinterpret_cast<void*>(&cuGetProcAddress_v2);
    return original;
}
extern "C" void* dlsym(void* handle,const char* name) noexcept {return substitute(name,real_lookup()(handle,name));}
extern "C" CUresult CUDAAPI cuGetProcAddress(const char* symbol,void** out,int version,cuuint64_t flags){
    using Fn=CUresult(CUDAAPI*)(const char*,void**,int,cuuint64_t);
    static auto fn=reinterpret_cast<Fn>(real_lookup()(RTLD_NEXT,"cuGetProcAddress"));CUresult c=fn(symbol,out,version,flags);
    if(c==CUDA_SUCCESS && !std::strcmp(symbol,"cuLaunchKernel"))*out=reinterpret_cast<void*>((flags&CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)?&cuLaunchKernel_ptsz:&cuLaunchKernel);
    return c;
}
extern "C" CUresult CUDAAPI cuGetProcAddress_v2(const char* symbol,void** out,int version,cuuint64_t flags,CUdriverProcAddressQueryResult* status){
    using Fn=CUresult(CUDAAPI*)(const char*,void**,int,cuuint64_t,CUdriverProcAddressQueryResult*);
    static auto fn=reinterpret_cast<Fn>(real_lookup()(RTLD_NEXT,"cuGetProcAddress_v2"));CUresult c=fn(symbol,out,version,flags,status);
    if(c==CUDA_SUCCESS && *out && !std::strcmp(symbol,"cuLaunchKernel"))*out=reinterpret_cast<void*>((flags&CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)?&cuLaunchKernel_ptsz:&cuLaunchKernel);
    return c;
}
extern "C" int observer_subscribe(){return 0;}
extern "C" int observer_lifecycle_version(){return 1;}
extern "C" int observer_start(const char*,unsigned long long p,unsigned long long v,size_t n){
    std::lock_guard<std::mutex> guard(lock);
    // A stopped scene must not lend pointers, stream identity, or launch
    // counters to its successor. Keep prior errors available until stop returns.
    if(rs_runtime || rs_ids || enabled || module || saved || scratch || tet_cache || rejected || residual || residual_cache || precise_p || precise_v)return -10;
    if(!p || !v || !n)return -11;
    CUcontext next=nullptr;
    CUresult status=cuPointerGetAttribute(&next,CU_POINTER_ATTRIBUTE_CONTEXT,p);
    if(status!=CUDA_SUCCESS)return int(status);
    CUcontext velocity_context=nullptr;
    status=cuPointerGetAttribute(&velocity_context,CU_POINTER_ATTRIBUTE_CONTEXT,v);
    if(status!=CUDA_SUCCESS)return int(status);
    if(next!=velocity_context)return -12;
    context=next;positions=p;velocities=v;nodes=n;sb=0;tets=0;mode=0;coupled_damping=false;
    solver_stream=nullptr;stream_seen=false;
    errors=0;active_calls=0;context_calls=0;soft_calls=0;damping_calls=0;ordered_contact_calls=0;
    for(auto& count:counts)count=0;
    for(auto& count:residual_counts)count=0;
    trace_step=-1;trace_launch=0;audit_sample_step=4800;
    for(auto& done:audit_done)done=false;
    return 0;
}
extern "C" void observer_step(int){}
extern "C" int observer_coupled_damping(int active){
    if(!enabled || mode!=6 || damping_calls || (active!=0 && active!=1))return -1;
    coupled_damping=active!=0;return 0;
}
extern "C" int observer_trace_start(const char* path){
    if(!enabled || trace_file)return -1;
    CUresult c=cuCtxPushCurrent(context);if(c!=CUDA_SUCCESS)return int(c);
    trace_p.resize(nodes);trace_v.resize(nodes);trace_mass.resize(nodes);
    c=cuMemcpyDtoH(trace_p.data(),positions,nodes*16);
    CUcontext previous;cuCtxPopCurrent(&previous);if(c!=CUDA_SUCCESS)return int(c);
    for(size_t i=0;i<nodes;++i){if(!(trace_p[i].w>0) || !std::isfinite(trace_p[i].w))return -2;trace_mass[i]=1./trace_p[i].w;}
    trace_file=std::fopen(path,"wx");if(!trace_file)return -3;
    audit_prefix=std::string(path)+".inputs/";
    std::fprintf(trace_file,"step\tlaunch\tphase\tkernel\tpx\tpy\tpz\tcx\tcy\tcz\tke\tinvdt\tpartition\tyoungs\tpoisson\tdamping\tmodel\troot_mass_ratio_min\troot_mass_ratio_max\troot_pv_imass_mismatch\tlx\tly\tlz\n");return 0;
}
extern "C" int observer_angular_start(const char* path){
    if(!trace_file || angular_states)return -1;angular_states=std::fopen(path,"wx");if(!angular_states)return -2;
    if(mode>=8){
        residual_states=std::fopen((std::string(path)+".residual.f64").c_str(),"wx");
        auxiliary_states=std::fopen((std::string(path)+".auxiliary.f32").c_str(),"wx");
        trace_residual.resize((mode>=9?3:2)*nodes);
        if(mode>=9){velocity_residual_states=std::fopen((std::string(path)+".velocity-residual.f64").c_str(),"wx");if(!velocity_residual_states)return -4;}trace_auxiliary.resize(2*nodes);
        if(!residual_states || !auxiliary_states)return -3;
    }
    return 0;
}
extern "C" void observer_audit_step(int step){audit_sample_step=step;}
extern "C" void observer_trace_step(int step){trace_step=step;}
extern "C" unsigned observer_trace_stop(){trace_step=-1;if(velocity_residual_states){if(std::fclose(velocity_residual_states))++errors;velocity_residual_states=nullptr;}if(residual_states){if(std::fclose(residual_states))++errors;residual_states=nullptr;}if(auxiliary_states){if(std::fclose(auxiliary_states))++errors;auxiliary_states=nullptr;}if(trace_file){if(std::fclose(trace_file))++errors;trace_file=nullptr;}if(angular_states){if(std::fclose(angular_states))++errors;angular_states=nullptr;}return errors.load();}
extern "C" int observer_enable_trial(const char* ptx,int m){
    if(enabled || !nodes || (m!=1&&m!=2&&m!=3&&m!=4&&m!=5&&m!=6&&m!=7&&m!=8&&m!=9&&m!=11))return -1;
    CUresult c=cuCtxPushCurrent(context);if(c!=CUDA_SUCCESS)return int(c);
    std::vector<float4> p(nodes);c=cuMemcpyDtoH(p.data(),positions,nodes*16);
    if(c==CUDA_SUCCESS)for(const auto& x:p)if(!(x.w>0))c=CUDA_ERROR_NOT_SUPPORTED;
    if(c==CUDA_SUCCESS)c=cuModuleLoad(&module,ptx);
    if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&splitfn,module,"split_root_mass");
    if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&preparefn,module,"prepare_average");
    if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&mergefn,module,"merge_copy_velocity");
    if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&cachefn,module,m>=9?"cache_partition_precision":"cache_partition_inputs");
    if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&dampingfn,module,m>=9?"update_partition_precision":"update_partition_damping");
    if(c==CUDA_SUCCESS)c=cuMemAlloc(&rejected,4);
    if(c==CUDA_SUCCESS)c=cuMemsetD32(rejected,0,1);
    if(c==CUDA_SUCCESS)c=cuMemAlloc(&saved,nodes*16);
    if(c==CUDA_SUCCESS)c=cuMemAlloc(&scratch,nodes*48);
    if(c==CUDA_SUCCESS && m>=8)c=cuModuleGetFunction(&capturefn,module,m>=9?"precision_capture":"residual_capture");
    if(c==CUDA_SUCCESS && m>=8)c=cuModuleGetFunction(&publishfn,module,m>=9?"precision_publish":"residual_publish");
    if(c==CUDA_SUCCESS && m>=8)c=cuModuleGetFunction(&residualmergefn,module,m>=9?"merge_copy_precision":"merge_copy_velocity_residual");
    if(c==CUDA_SUCCESS && m>=8)c=cuMemAlloc(&residual,nodes*(m>=9?72:48));
    if(c==CUDA_SUCCESS && m>=8)c=cuMemsetD8(residual,0,nodes*(m>=9?72:48));
    if(c==CUDA_SUCCESS && m>=8)c=cuMemAlloc(&residual_cache,nodes*(m>=9?64:48));
    CUcontext previous;cuCtxPopCurrent(&previous);if(c==CUDA_SUCCESS){mode=m;enabled=true;}return int(c);
}
extern "C" unsigned long long observer_trial_count(int kind){return kind>=0&&kind<3?counts[kind]:kind==3?active_calls.load():kind==4?context_calls.load():kind==5?soft_calls.load():kind==6?damping_calls:(kind>=7 && kind<=9)?residual_counts[kind-7]:kind==11?ordered_contact_calls:0;}
extern "C" unsigned observer_stop(){rs_stop();pc_stop();observer_articulation_work_stop();observer_angular_transport_stop();block_stop();observer_contact_policy_stop();observer_contact_stop();observer_trace_stop();enabled=false;if(!context)return errors.load();if(cuCtxPushCurrent(context)!=CUDA_SUCCESS)return ++errors;if(cuCtxSynchronize()!=CUDA_SUCCESS)++errors;if(rejected){unsigned n=0;if(cuMemcpyDtoH(&n,rejected,4)!=CUDA_SUCCESS)++errors;else errors+=n;cuMemFree(rejected);rejected=0;}if(tet_cache){cuMemFree(tet_cache);tet_cache=0;}if(saved)cuMemFree(saved);if(scratch)cuMemFree(scratch);if(precise_p)cuMemFree(precise_p);if(precise_v)cuMemFree(precise_v);precise_p=precise_v=0;if(residual)cuMemFree(residual);if(residual_cache)cuMemFree(residual_cache);residual=residual_cache=0;friction_free();normal_free();rr_free();if(module)cuModuleUnload(module);saved=scratch=0;module=nullptr;CUcontext previous;cuCtxPopCurrent(&previous);return errors.load();}
#include "physx_damping_audit.inc"

extern "C" int observer_read_residual(double* out,size_t n){
    if(!enabled || mode<8 || n!=nodes || !out)return -1;
    CUresult c=cuCtxPushCurrent(context);if(c!=CUDA_SUCCESS)return int(c);
    c=cuCtxSynchronize();if(c==CUDA_SUCCESS)c=cuMemcpyDtoH(out,residual,nodes*48);
    CUcontext previous;cuCtxPopCurrent(&previous);return int(c);
}

extern "C" int observer_read_velocity_residual(double* out,size_t n){
    if(!enabled || mode<9 || n!=nodes || !out)return -1;
    CUresult c=cuCtxPushCurrent(context);if(c!=CUDA_SUCCESS)return int(c);
    c=cuCtxSynchronize();if(c==CUDA_SUCCESS)c=cuMemcpyDtoH(out,residual+nodes*48,nodes*24);
    CUcontext previous;cuCtxPopCurrent(&previous);return int(c);
}
