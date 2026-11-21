// Explicit observer activation; output belongs to the current cloth lifecycle.
static bool fht_enabled=false;static FILE* fht_file=nullptr;
static unsigned fht_target_count=0;static unsigned long long fht_target_keys[2]{},fht_last[6]{};
static void fht_reset(){
 if(fht_file){std::fclose(fht_file);fht_file=nullptr;}
 fht_enabled=false;fht_target_count=0;for(auto& x:fht_target_keys)x=0;for(auto& x:fht_last)x=0;
}
static bool fht_bind(){
 if(!fht_enabled||!fht_file)return false;
 for(unsigned k=0;k<fht_target_count;++k)for(unsigned shift:{0u,16u,32u,48u})if(((fht_target_keys[k]>>shift)&65535)>=cf_n)return false;
 CUdeviceptr address=0;size_t bytes=0;
 return cuModuleGetGlobal(&address,&bytes,cf_module,"fht_keys")==CUDA_SUCCESS&&bytes==16&&cuMemcpyHtoD(address,fht_target_keys,16)==CUDA_SUCCESS;
}
static bool fht_flush(unsigned epoch){
 if(!fht_enabled)return true;bool ok=fht_file&&cf_module&&epoch<=131072;
 fht_last[0]=epoch;fht_last[1]=fht_target_count;
 CUdeviceptr address=0;size_t bytes=0;
 if(ok)ok=cuModuleGetGlobal(&address,&bytes,cf_module,"fht_counts")==CUDA_SUCCESS&&bytes==24&&cf_read(fht_last+2,address,24);
 std::vector<unsigned> data;
 if(ok){
  data.resize(size_t(epoch)*2*48);
  ok=cuModuleGetGlobal(&address,&bytes,cf_module,"fht_records")==CUDA_SUCCESS&&bytes==size_t(131072)*2*48*4&&cf_read(data.data(),address,data.size()*4);
 }
 if(ok){
  unsigned header[12]={0x46544831,1,fht_target_count,131072,epoch,48,unsigned(fht_last[2]),unsigned(fht_last[3]+fht_last[4]),unsigned(fht_target_keys[0]),unsigned(fht_target_keys[0]>>32),unsigned(fht_target_keys[1]),unsigned(fht_target_keys[1]>>32)};
  ok=std::fwrite(header,4,12,fht_file)==12&&std::fwrite(data.data(),4,data.size(),fht_file)==data.size();
 }
 if(fht_file){if(std::fclose(fht_file))ok=false;fht_file=nullptr;}
 fht_last[5]=ok?0:1;return ok;
}
