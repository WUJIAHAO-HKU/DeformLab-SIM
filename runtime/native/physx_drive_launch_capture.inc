// Optional read-only capture for the exact native Cartesian drive kernel.
// Included only in the separately built observation library.
namespace {
FILE* dl_index=nullptr;FILE* dl_blob=nullptr;unsigned long long dl_records=0;
unsigned long long dl_stop(){
 if(dl_index){if(std::fclose(dl_index))++errors;dl_index=nullptr;}
 if(dl_blob){if(std::fclose(dl_blob))++errors;dl_blob=nullptr;}
 return dl_records;
}
bool dl_selected(const char* name){return dl_index && !std::strcmp(name,"artiSolveInternalConstraintsTGS1T");}
bool dl_snapshot(void** params,const char* phase,unsigned long long seq,
 unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned shared){
 if(!dl_index || !dl_blob || !params || ct_step<0 || !enabled || !nr_module)return false;
 CUcontext current=nullptr;
 if(cuCtxGetCurrent(&current)!=CUDA_SUCCESS || current!=context || cuCtxSynchronize()!=CUDA_SUCCESS)return false;
 // Sizes taken from the binary-matched PTX .entry, never inferred from values.
 const unsigned sizes[15]={8,4,4,1,4,4,4,8,8,8,1,1,1,4,1};
 unsigned char args[15][8]={};
 for(unsigned i=0;i<15;++i){if(!params[i])return false;std::memcpy(args[i],params[i],sizes[i]);}
 unsigned char core[344],blocks[7168];
 if(cuMemcpyDtoH(core,field<CUdeviceptr>(args[0],0),sizeof(core))!=CUDA_SUCCESS)return false;
 if(field<unsigned>(core,44)!=1 || field<unsigned>(core,248)!=4 || field<unsigned>(core,252)!=3)return false;
 if(cuMemcpyDtoH(blocks,field<CUdeviceptr>(core,128),sizeof(blocks))!=CUDA_SUCCESS)return false;
 // General joints, tendons and rigid self/static constraints need more buffers.
 for(unsigned off:{3840u,3968u,4096u,7072u})if(field<unsigned>(blocks,off)!=0)return false;
 std::vector<unsigned char> links(4*29184),stack(4*1664),shared_desc(224);
 if(cuMemcpyDtoH(links.data(),field<CUdeviceptr>(core,136),links.size())!=CUDA_SUCCESS)return false;
 for(unsigned i=0;i<4;++i)for(unsigned off:{22016u,22272u})if(field<unsigned>(links.data()+i*29184,off)!=0)return false;
 if(cuMemcpyDtoH(stack.data(),field<CUdeviceptr>(core,144),stack.size())!=CUDA_SUCCESS)return false;
 auto shared_pointer=field<CUdeviceptr>(args[9],0);
 if(shared_pointer && cuMemcpyDtoH(shared_desc.data(),shared_pointer,shared_desc.size())!=CUDA_SUCCESS)return false;
 std::fprintf(dl_index,"{\"seq\":%llu,\"step\":%d,\"phase\":\"%s\",\"kernel\":\"artiSolveInternalConstraintsTGS1T\",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"shared_bytes\":%u,\"parameter_sizes\":[8,4,4,1,4,4,4,8,8,8,1,1,1,4,1],\"parameter_hex\":[",seq,ct_step.load(),phase,gx,gy,gz,bx,by,bz,shared);
 for(unsigned i=0;i<15;++i){if(i)std::fputc(',',dl_index);std::fputc('"',dl_index);for(unsigned j=0;j<sizes[i];++j)std::fprintf(dl_index,"%02x",args[i][j]);std::fputc('"',dl_index);}
 std::fprintf(dl_index,"],\"sections\":{");bool first=true,ok=true;
 auto save=[&](const char* key,const void* data,size_t size){
  long offset=std::ftell(dl_blob);if(offset<0 || (size && std::fwrite(data,1,size,dl_blob)!=size)){ok=false;return;}
  if(!first)std::fputc(',',dl_index);first=false;std::fprintf(dl_index,"\"%s\":[%ld,%zu]",key,offset,size);
 };
 save("core",core,sizeof(core));save("traversal_stack",stack.data(),stack.size());
 save("shared_desc",shared_desc.data(),shared_pointer?shared_desc.size():0);
 std::fprintf(dl_index,"}}\n");++dl_records;return ok && !std::ferror(dl_index);
}
}
extern "C" int observer_drive_launch_start(const char* path){
 if(!path || !*path || !enabled || mode!=6 || !nr_module || dl_index || dl_blob)return -1;
 dl_index=std::fopen((std::string(path)+".jsonl").c_str(),"wx");
 if(!dl_index)return -2;
 dl_blob=std::fopen((std::string(path)+".bin").c_str(),"wx");
 if(!dl_blob){std::fclose(dl_index);dl_index=nullptr;return -2;}
 dl_records=0;return 0;
}
extern "C" unsigned long long observer_drive_launch_stop(){return dl_stop();}
