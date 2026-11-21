// Diagnostic launch-boundary sampling of the running binary, never a state repair.
#include <cuda.h>
#include <cupti.h>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {
CUpti_SubscriberHandle subscriber;
bool subscribed=false;
CUdeviceptr positions=0, velocities=0;
CUcontext target_context=nullptr;
size_t nodes=0, snapshots=0;
FILE *events=nullptr,*states=nullptr,*extra_states=nullptr;
CUdeviceptr soft_struct=0,rigid_desc=0,shared_desc=0;
std::atomic<int> active_step{-1};
std::atomic<unsigned> errors{0};
std::mutex output_lock;
thread_local bool inside=false;
std::string quoted(const char* s) {
    std::string r="\"";
    for (;s && *s;++s) {if (*s=='"'||*s=='\\') r+='\\'; if(static_cast<unsigned char>(*s)>=32)r+=*s;}
    return r+'"';
}
template<class T> T field(const unsigned char* bytes,size_t offset) {T value;std::memcpy(&value,bytes+offset,sizeof(T));return value;}
// Offsets below come from DWARF in the very same loaded GPU binary (SHA-256
// 05220192126a9bacedeb432eca6070be2faddc5879475b90be00bfeb815c1cc7).
// Python refuses any other binary before enabling this private-layout diagnostic.
void extras(const cuLaunchKernel_params* args,const char* kernel) {
    bool first=true;
    auto dump=[&](const char* name,CUdeviceptr pointer,size_t bytes) {
        if(!pointer || !bytes || bytes>4000000){++errors;return;}
        std::vector<unsigned char> data(bytes);
        CUresult status=cuMemcpyDtoH(data.data(),pointer,bytes);
        if(status!=CUDA_SUCCESS)++errors;
        if(!first)fputc(',',events);first=false;
        fprintf(events,"{\"name\":%s,\"offset\":%ld,\"bytes\":%zu,\"cuda_status\":%d}",quoted(name).c_str(),ftell(extra_states),bytes,int(status));
        if(fwrite(data.data(),1,bytes,extra_states)!=bytes)++errors;
    };
    if(!std::strcmp(kernel,"sb_gm_preIntegrateLaunch")) {
        unsigned id=0;
        auto base=*static_cast<const CUdeviceptr*>(args->kernelParams[0]);
        auto active=*static_cast<const CUdeviceptr*>(args->kernelParams[1]);
        if(cuMemcpyDtoH(&id,active,4)==CUDA_SUCCESS && id<10000)soft_struct=base+id*432;
        else ++errors;
    }
    if(!std::strcmp(kernel,"sb_solveRigidSoftCollisionLaunchTGS")) {
        rigid_desc=*static_cast<const CUdeviceptr*>(args->kernelParams[5]);
        shared_desc=*static_cast<const CUdeviceptr*>(args->kernelParams[7]);
        unsigned contacts=0;
        auto countptr=*static_cast<const CUdeviceptr*>(args->kernelParams[3]);
        if(cuMemcpyDtoH(&contacts,countptr,4)!=CUDA_SUCCESS || contacts>10000)++errors;
        else if(contacts) {
            dump("rigid_contact_delta",*static_cast<const CUdeviceptr*>(args->kernelParams[8]),contacts*32);
            dump("contact_applied",*static_cast<const CUdeviceptr*>(args->kernelParams[9]),contacts*4);
        }
    }
    if(soft_struct && !std::strncmp(kernel,"sb_",3)) {
        unsigned char sb[432];
        if(cuMemcpyDtoH(sb,soft_struct,sizeof(sb))!=CUDA_SUCCESS){++errors;return;}
        if(field<CUdeviceptr>(sb,88)!=velocities || field<CUdeviceptr>(sb,96)!=positions || field<unsigned>(sb,372)!=nodes){++errors;return;}
        dump("soft_delta_position",field<CUdeviceptr>(sb,128),nodes*16);
        dump("soft_contact_delta",field<CUdeviceptr>(sb,272),nodes*16);
        if(!std::strcmp(kernel,"sb_gm_cp_solveTetrahedronsPartitionLaunch") || !std::strcmp(kernel,"sb_gm_cp_averageVertsLaunch")) {
            unsigned copies=0;auto acc=field<CUdeviceptr>(sb,200);auto tets=field<unsigned>(sb,376);
            if(cuMemcpyDtoH(&copies,acc+(nodes-1)*4,4)!=CUDA_SUCCESS || tets>10000 || copies>100000 || field<unsigned>(sb,412)!=1){++errors;return;}
            dump("cp_positions",field<CUdeviceptr>(sb,184),(4*tets+copies)*16);
            dump("cp_velocities",field<CUdeviceptr>(sb,176),(4*tets+copies)*16);
            dump("cp_pull_indices",field<CUdeviceptr>(sb,400),tets*16);
            dump("cp_accumulated_copies",acc,nodes*4);
            dump("cp_remap",field<CUdeviceptr>(sb,192),tets*16);
            dump("cp_partitions",field<CUdeviceptr>(sb,208),field<unsigned>(sb,380)*4);
            dump("cp_ordered_tets",field<CUdeviceptr>(sb,152),tets*4);
            dump("simulation_tets",field<CUdeviceptr>(sb,80),tets*16);
        }
    }
    if(rigid_desc) {
        unsigned char desc[304];
        if(cuMemcpyDtoH(desc,rigid_desc,sizeof(desc))!=CUDA_SUCCESS){++errors;return;}
        unsigned count=field<unsigned>(desc,268);
        if(count>64){++errors;return;}
        if(count) {
            dump("rigid_solver_velocity",field<CUdeviceptr>(desc,0),count*32);
            dump("rigid_solver_body_data",field<CUdeviceptr>(desc,16),count*96);
            if(shared_desc) {
                unsigned char shared[224];
                if(cuMemcpyDtoH(shared,shared_desc,sizeof(shared))!=CUDA_SUCCESS){++errors;return;}
                // TGS reads the accumulated slab, packed as 3 float4s per body.
                auto offset=field<unsigned>(desc,276);
                if(offset>100000){++errors;return;}
                dump("rigid_tgs_accumulated_velocity",field<CUdeviceptr>(shared,160)+size_t(offset)*16,count*48);
            }
        }
    }
}
void CUPTIAPI callback(void*,CUpti_CallbackDomain domain,CUpti_CallbackId id,const void* raw) {
    if(inside || domain!=CUPTI_CB_DOMAIN_DRIVER_API || active_step.load()<0)return;
    auto* d=static_cast<const CUpti_CallbackData*>(raw);
    if(d->context!=target_context)return;
    if(id!=CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel && id!=CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz) {
        if(d->callbackSite==CUPTI_API_ENTER) {
            std::lock_guard<std::mutex> guard(output_lock);
            fprintf(events,"{\"step\":%d,\"unsupported_launch\":%s}\n",active_step.load(),quoted(d->functionName).c_str());
            ++errors;
        }
        return;
    }
    inside=true;
    auto* args=static_cast<const cuLaunchKernel_params*>(d->functionParams);
    // Capture is rejected rather than implicitly synchronizing an active CUDA graph.
    CUstreamCaptureStatus capture;
    CUresult status=cuStreamIsCapturing(args->hStream,&capture);
    if(status==CUDA_SUCCESS && capture!=CU_STREAM_CAPTURE_STATUS_NONE)status=CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    if(status==CUDA_SUCCESS)status=cuCtxSynchronize();
    std::vector<float> data(nodes*8);
    if(status==CUDA_SUCCESS)status=cuMemcpyDtoH(data.data(),positions,nodes*16);
    if(status==CUDA_SUCCESS)status=cuMemcpyDtoH(data.data()+nodes*4,velocities,nodes*16);
    std::lock_guard<std::mutex> guard(output_lock);
    if(status!=CUDA_SUCCESS)++errors;
    auto name=quoted(d->symbolName ? d->symbolName : "<unknown>");
    fprintf(events,"{\"step\":%d,\"site\":%u,\"kernel\":%s,\"correlation\":%u,\"context\":%u,\"stream\":%llu,\"cuda_status\":%d,\"snapshot\":%zu,\"parameters\":[",
        active_step.load(),unsigned(d->callbackSite),name.c_str(),d->correlationId,d->contextUid,
        (unsigned long long)args->hStream,int(status),snapshots);
    if(d->callbackSite==CUPTI_API_ENTER && args->kernelParams) {
        for(size_t i=0;i<64;++i) {
            size_t offset=0,size=0;
            if(cuFuncGetParamInfo(args->f,i,&offset,&size)!=CUDA_SUCCESS)break;
            if(i)fputc(',',events);
            fprintf(events,"{\"index\":%zu,\"offset\":%zu,\"size\":%zu,\"hex\":\"",i,offset,size);
            if(size<=256 && args->kernelParams[i]) {
                auto* bytes=static_cast<const unsigned char*>(args->kernelParams[i]);
                for(size_t j=0;j<size;++j)fprintf(events,"%02x",bytes[j]);
            }
            fprintf(events,"\"}");
        }
    }
    fprintf(events,"],\"extra\":[");
    if(status==CUDA_SUCCESS && args->kernelParams)extras(args,d->symbolName ? d->symbolName : "");
    fprintf(events,"]}\n");
    if(fwrite(data.data(),sizeof(float),data.size(),states)!=data.size())++errors;
    ++snapshots;
    inside=false;
}
}
extern "C" int observer_subscribe() {
    if(subscribed)return 0;
    CUptiResult c=cuptiSubscribe(&subscriber,callback,nullptr);
    if(c!=CUPTI_SUCCESS)return 2000+int(c);
    subscribed=true;
    for(auto id:{CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz,
        CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel,CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx,
        CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch}) {
        c=cuptiEnableCallback(1,subscriber,CUPTI_CB_DOMAIN_DRIVER_API,id);
        if(c!=CUPTI_SUCCESS)return 2000+int(c);
    }
    return 0;
}
extern "C" int observer_start(const char* directory,unsigned long long pos,unsigned long long vel,size_t count) {
    if(events || !count || !subscribed)return -1;
    positions=pos;velocities=vel;nodes=count;
    CUresult r=cuPointerGetAttribute(&target_context,CU_POINTER_ATTRIBUTE_CONTEXT,positions);
    if(r!=CUDA_SUCCESS)return 1000+int(r);
    events=fopen((std::string(directory)+"/kernel-events.jsonl").c_str(),"wx");
    states=fopen((std::string(directory)+"/kernel-states.f32").c_str(),"wx");
    extra_states=fopen((std::string(directory)+"/kernel-extra.bin").c_str(),"wx");
    if(!events||!states||!extra_states)return -2;
    return 0;
}
extern "C" void observer_step(int step) {active_step.store(step);}
extern "C" unsigned observer_stop() {
    active_step.store(-1);
    cuptiUnsubscribe(subscriber);
    subscribed=false;
    if(events){fclose(events);events=nullptr;}
    if(states){fclose(states);states=nullptr;}
    if(extra_states){fclose(extra_states);extra_states=nullptr;}
    return errors.load();
}
