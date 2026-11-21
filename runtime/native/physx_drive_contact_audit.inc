// Separate read-only evidence for the added correction. Existing articulation
// observations retain their original names and invocation keys.
namespace {
FILE* dc_audit_index=nullptr;FILE* dc_audit_blob=nullptr;unsigned long long dc_audit_records=0;
void dc_audit_stop(){
 if(dc_audit_index){if(std::fclose(dc_audit_index))++errors;dc_audit_index=nullptr;}
 if(dc_audit_blob){if(std::fclose(dc_audit_blob))++errors;dc_audit_blob=nullptr;}
}
bool dc_observe(const char* phase){
 if(!dc_audit_index || ct_step<0)return true;
 if(!enabled || !dc_active() || !dc_pending || !dc_have_drive || cuCtxSynchronize()!=CUDA_SUCCESS)return false;
 CUcontext current=nullptr;if(cuCtxGetCurrent(&current)!=CUDA_SUCCESS||current!=context)return false;
 unsigned char core[344];if(cuMemcpyDtoH(core,dc_values[0],344)!=CUDA_SUCCESS)return false;
 if(field<unsigned>(core,44)!=1 || field<unsigned>(core,248)!=4 || field<unsigned>(core,252)!=3)return false;
 std::fprintf(dc_audit_index,"{\"step\":%d,\"call\":%llu,\"phase\":\"%s\",\"velocity_pass\":%u,\"sections\":{",ct_step.load(),dc_contacts,phase,unsigned(static_cast<unsigned char>(dc_values[3])));
 bool first=true,ok=true;
 auto save=[&](const char* key,const void* data,size_t size){
  long offset=std::ftell(dc_audit_blob);if(offset<0 || (size&&std::fwrite(data,1,size,dc_audit_blob)!=size)){ok=false;return;}
  if(!first)std::fprintf(dc_audit_index,",");first=false;std::fprintf(dc_audit_index,"\"%s\":[%ld,%zu]",key,offset,size);
 };
 auto gpu=[&](const char* key,CUdeviceptr address,size_t size){std::vector<unsigned char> bytes(size);if(cuMemcpyDtoH(bytes.data(),address,size)!=CUDA_SUCCESS){ok=false;return;}save(key,bytes.data(),size);};
 save("core",core,344);gpu("blocks",field<CUdeviceptr>(core,128),7168);gpu("link_blocks",field<CUdeviceptr>(core,136),4*29184);
 gpu("dof_blocks",field<CUdeviceptr>(core,184),3*12160);gpu("coupling_state",dc_state,8*sizeof(double));gpu("drive_delta",dc_delta,3*sizeof(float));
 std::fprintf(dc_audit_index,"},\"parameter_hex\":[");
 for(unsigned i=0;i<15;++i){if(i)std::fprintf(dc_audit_index,",");std::fprintf(dc_audit_index,"\"");for(unsigned j=0;j<dc_sizes[i];++j)std::fprintf(dc_audit_index,"%02x",reinterpret_cast<const unsigned char*>(&dc_values[i])[j]);std::fprintf(dc_audit_index,"\"");}
 std::fprintf(dc_audit_index,"]}\n");++dc_audit_records;return ok&&!std::ferror(dc_audit_index);
}
}
extern "C" int observer_drive_contact_audit_start(const char* prefix){
 if(!prefix||!enabled||!dc_active()||!ct_coupon||dc_audit_index||dc_audit_blob)return -1;
 dc_audit_records=0;dc_audit_index=std::fopen((std::string(prefix)+".jsonl").c_str(),"wx");
 if(!dc_audit_index)return -2;
 dc_audit_blob=std::fopen((std::string(prefix)+".bin").c_str(),"wx");
 if(!dc_audit_blob){dc_audit_stop();return -2;}
 return 0;
}
extern "C" unsigned long long observer_drive_contact_audit_stop(){dc_audit_stop();return dc_audit_records;}
