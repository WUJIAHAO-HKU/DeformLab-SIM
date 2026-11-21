// Diagnostic-only all-feature ring; never writes solver arrays.
static constexpr unsigned fhr_slots=128;
static FILE* fhr_file=nullptr;static bool fhr_enabled=false;
static CUdeviceptr fhr_metadata=0,fhr_positions=0,fhr_roots=0,fhr_entries=0,fhr_counts=0,fhr_rejection=0,fhr_decode=0;
static CUfunction fhr_capture=nullptr,fhr_publish=nullptr;
static unsigned fhr_nodes=0,fhr_capacity=0;static unsigned long long fhr_last[9]{};
static void fhr_reset(){if(fhr_file){std::fclose(fhr_file);fhr_file=nullptr;}fhr_enabled=false;fhr_nodes=fhr_capacity=0;for(auto& v:fhr_last)v=0;}
static bool fhr_start(const char* path){
 if(fhr_enabled||!path)return false;std::string name=std::string(path)+".ring.bin";
 fhr_file=std::fopen(name.c_str(),"wx");if(!fhr_file)return false;fhr_enabled=true;return true;
}
static bool fhr_allocate(unsigned nodes,unsigned capacity){
 if(!fhr_enabled||!fhr_file||!nodes||nodes>4096||!capacity||capacity>32768)return false;
 fhr_nodes=nodes;fhr_capacity=capacity;
 for(auto item:{std::pair<CUdeviceptr*,size_t>{&fhr_metadata,size_t(fhr_slots)*32},{&fhr_positions,size_t(fhr_slots)*nodes*16},{&fhr_roots,size_t(fhr_slots)*nodes*16},{&fhr_entries,size_t(fhr_slots)*capacity*24},{&fhr_counts,32}}){if(cuMemAlloc(item.first,item.second)!=CUDA_SUCCESS||cuMemsetD8(*item.first,0,item.second)!=CUDA_SUCCESS)return false;}
 size_t bytes=0;
 return cuModuleGetFunction(&fhr_capture,cf_module,"cloth_history_ring_capture")==CUDA_SUCCESS&&cuModuleGetFunction(&fhr_publish,cf_module,"cloth_history_ring_publish")==CUDA_SUCCESS&&cuModuleGetGlobal(&fhr_rejection,&bytes,cf_module,"cloth_history_first_failure")==CUDA_SUCCESS&&bytes==512&&cuModuleGetGlobal(&fhr_decode,&bytes,cf_module,"cloth_history_first_decode")==CUDA_SUCCESS&&bytes==64;
}
static bool fhr_step(Launch original,CUstream stream,CUdeviceptr current,CUdeviceptr previous,CUdeviceptr history,unsigned epoch,unsigned long long frame){
 if(!fhr_enabled||!fhr_metadata)return false;
 unsigned slots=fhr_slots;
 void* capture[]={&current,&previous,&history,&fhr_nodes,&fhr_capacity,&slots,&epoch,&frame,&fhr_rejection,&fhr_decode,&fhr_metadata,&fhr_positions,&fhr_roots,&fhr_entries,&fhr_counts};
 if(cf_launch(original,fhr_capture,(std::max(fhr_nodes,fhr_capacity)+255)/256,256,stream,capture)!=CUDA_SUCCESS)return false;
 void* publish[]={&fhr_nodes,&fhr_capacity,&slots,&epoch,&fhr_rejection,&fhr_decode,&fhr_metadata,&fhr_counts};
 return cf_launch(original,fhr_publish,1,1,stream,publish)==CUDA_SUCCESS;
}
static bool fhr_flush(unsigned epoch){
 if(!fhr_enabled)return true;
 bool ok=fhr_file&&fhr_metadata&&fhr_nodes&&fhr_capacity;
 unsigned rejection[128]{},decode[16]{};unsigned long long counts[4]{};
 if(ok)ok=cf_read(rejection,fhr_rejection,512)&&cf_read(decode,fhr_decode,64)&&cf_read(counts,fhr_counts,32);
 fhr_last[0]=fhr_slots;fhr_last[1]=fhr_nodes;fhr_last[2]=fhr_capacity;for(unsigned i=0;i<4;++i)fhr_last[3+i]=counts[i];
 fhr_last[7]=size_t(fhr_slots)*(32+size_t(fhr_nodes)*32+size_t(fhr_capacity)*24)+32;
 if(ok){
  unsigned header[16]={0x46524831,1,fhr_slots,fhr_nodes,fhr_capacity,6,8,4,epoch,rejection[0]==2?rejection[4]:0,decode[0]==2?decode[4]:0,0,0,0,0,0};
  ok=std::fwrite(header,4,16,fhr_file)==16&&std::fwrite(counts,8,4,fhr_file)==4;
  for(auto item:{std::pair<CUdeviceptr,size_t>{fhr_metadata,size_t(fhr_slots)*32},{fhr_positions,size_t(fhr_slots)*fhr_nodes*16},{fhr_roots,size_t(fhr_slots)*fhr_nodes*16},{fhr_entries,size_t(fhr_slots)*fhr_capacity*24}}){
   if(!ok)break;std::vector<unsigned char> data(item.second);ok=cf_read(data.data(),item.first,item.second)&&std::fwrite(data.data(),1,item.second,fhr_file)==item.second;
  }
 }
 if(fhr_file){if(std::fclose(fhr_file))ok=false;fhr_file=nullptr;}
 fhr_last[8]=ok?0:1;return ok;
}
static bool fhr_release(){
 for(CUdeviceptr* p:{&fhr_metadata,&fhr_positions,&fhr_roots,&fhr_entries,&fhr_counts})if(*p){if(cuMemFree(*p)!=CUDA_SUCCESS)return false;*p=0;}
 fhr_rejection=fhr_decode=0;return true;
}
