// Exact-binary read-only CP input/output audit. Included by the host interposer.
static void audit_cp_partition(CUstream stream,void** params,bool after){
    if(!trace_file || trace_step.load()!=audit_sample_step || audit_prefix.empty())return;
    unsigned part=*static_cast<unsigned*>(params[4]);if(part>=8 || audit_done[part])return;
    if(cuStreamSynchronize(stream)!=CUDA_SUCCESS){++errors;return;}
    unsigned char data[432];if(cuMemcpyDtoH(data,sb,sizeof(data))!=CUDA_SUCCESS){++errors;return;}
    unsigned nt=field<unsigned>(data,376),np=field<unsigned>(data,380),nc=0;
    if(cuMemcpyDtoH(&nc,field<CUdeviceptr>(data,200)+(nodes-1)*4,4)!=CUDA_SUCCESS){++errors;return;}
    auto prefix=audit_prefix+"partition-"+std::to_string(part)+(after?"-after-":"-before-");
    auto dump=[&](const char* name,CUdeviceptr pointer,size_t bytes){
        std::vector<unsigned char> buffer(bytes);
        if(cuMemcpyDtoH(buffer.data(),pointer,bytes)!=CUDA_SUCCESS){++errors;return;}
        FILE* file=std::fopen((prefix+name+".bin").c_str(),"wb");
        if(!file){++errors;return;}if(std::fwrite(buffer.data(),1,bytes,file)!=bytes)++errors;if(std::fclose(file))++errors;
    };
    dump("root_pos",positions,nodes*16);dump("root_vel",velocities,nodes*16);
    dump("cp_pos",field<CUdeviceptr>(data,184),(4*nt+nc)*16);
    dump("cp_vel",field<CUdeviceptr>(data,176),(4*nt+nc)*16);
    dump("pull",field<CUdeviceptr>(data,400),nt*16);dump("remap",field<CUdeviceptr>(data,192),nt*16);
    dump("counts",field<CUdeviceptr>(data,200),nodes*4);dump("partitions",field<CUdeviceptr>(data,208),np*4);
    dump("rest",field<CUdeviceptr>(data,136),((nt+31)/32)*1152);
    dump("rotation",field<CUdeviceptr>(data,168),nt*16);
    dump("material_indices",field<CUdeviceptr>(data,288),nt*2);
    std::vector<unsigned short> ids(nt);
    if(cuMemcpyDtoH(ids.data(),field<CUdeviceptr>(data,288),nt*2)!=CUDA_SUCCESS){++errors;return;}
    unsigned highest=0;for(auto id:ids)if(id>highest)highest=id;
    // The exact embedded PTX uses a 32-byte stride and offsets 0,4,12,16.
    auto materials=*static_cast<CUdeviceptr*>(params[7]);dump("materials",materials,(highest+1)*32);
    unsigned char mat[32];if(cuMemcpyDtoH(mat,materials+ids[0]*32,32)!=CUDA_SUCCESS){++errors;return;}
    material_youngs=field<float>(mat,0);material_poisson=field<float>(mat,4);
    material_damping=field<float>(mat,12);material_model=field<unsigned short>(mat,16);
    FILE* meta=std::fopen((prefix+"meta.json").c_str(),"wx");
    if(!meta){++errors;return;}
    std::fprintf(meta,"{\"nodes\":%zu,\"tets\":%u,\"copies\":%u,\"partitions\":%u,\"invdt\":%.9g,\"tgs\":%u,\"first_iteration\":%u}\n",nodes,nt,nc,np,*static_cast<float*>(params[3]),unsigned(*static_cast<unsigned char*>(params[5])),unsigned(*static_cast<unsigned char*>(params[6])));
    if(std::fclose(meta))++errors;
    if(after)audit_done[part]=true;
}
