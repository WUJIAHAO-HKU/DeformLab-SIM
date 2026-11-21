// Bounded read-only snapshots: first invalid edge, latest edge, rejected input.
bool ct_enabled=false,ct_ready=false,ct_pending=false,ct_rejection_saved=false;
unsigned ct_invalid_mask=0;
std::string ct_prefix,ct_name,ct_edge_name;
CUfunction ct_check=nullptr;
struct CTBody{CUdeviceptr before=0,edge_before=0,edge_after=0,result=0;};CTBody ct_body[2];
unsigned long long ct_observations=0,ct_edges=0,ct_snapshots=0,ct_errors=0,ct_edge_step=0,ct_edge_call=0;
bool ct_fail(const char* why){++ct_errors;++errors;std::fprintf(stderr,"Cone trace error: %s\n",why);return false;}
bool ct_copy_current(CUdeviceptr to,unsigned j){
 auto& b=cg_body[j];CUdeviceptr from[]={b.p,b.v,b.delta,ss_bodies[j].pending};
 for(unsigned k=0;k<4;++k)if(cuMemcpyDtoD(to+size_t(k)*b.n*16,from[k],size_t(b.n)*16)!=CUDA_SUCCESS)return false;return true;
}
bool ct_initialize(){
 if(ct_ready)return true;if(!cg_module)return true;
 if(cuModuleGetFunction(&ct_check,cg_module,"cone_trace_check")!=CUDA_SUCCESS)return ct_fail("check symbol");
 for(unsigned j=0;j<2;++j){auto& c=ct_body[j];size_t size=size_t(cg_body[j].n)*64;
  if(cuMemAlloc(&c.before,size)!=CUDA_SUCCESS||cuMemAlloc(&c.edge_before,size)!=CUDA_SUCCESS||cuMemAlloc(&c.edge_after,size)!=CUDA_SUCCESS||cuMemAlloc(&c.result,16)!=CUDA_SUCCESS)return ct_fail("trace allocation");
 }
 ct_ready=true;return true;
}
bool ct_dump(const char* suffix,bool edge){
 if(!ct_ready)return ct_fail("dump before binding");
 std::string stem=ct_prefix+"-"+suffix;
 FILE* binary=std::fopen((stem+".bin").c_str(),"wx");if(!binary)return ct_fail("exclusive snapshot create");
 std::ostringstream meta;meta<<"{\"kind\":"<<std::quoted(suffix)<<",\"step\":"<<ss_steps<<",\"guard_calls\":"<<cg_calls<<",\"edge_step\":"<<ct_edge_step<<",\"edge_guard_calls\":"<<ct_edge_call<<",\"edge_kernel\":"<<std::quoted(ct_edge_name)<<",\"last_observed_kernel\":"<<std::quoted(ct_name)<<",\"bodies\":[";
 size_t offset=0;bool okay=true;
 for(unsigned j=0;j<2&&okay;++j){
  auto& b=cg_body[j];auto& c=ct_body[j];size_t size=size_t(b.n)*64;
  std::vector<unsigned char> before(size),after(size),tets(size_t(b.nt)*16),rest(size_t(b.nt)*8);
  if(edge){okay=cuMemcpyDtoH(before.data(),c.edge_before,size)==CUDA_SUCCESS&&cuMemcpyDtoH(after.data(),c.edge_after,size)==CUDA_SUCCESS;}
  else{okay=cuMemcpyDtoH(before.data(),c.before,size)==CUDA_SUCCESS;CUdeviceptr ptrs[]={b.p,b.v,b.delta,ss_bodies[j].pending};for(unsigned k=0;k<4&&okay;++k)okay=cuMemcpyDtoH(after.data()+size_t(k)*b.n*16,ptrs[k],size_t(b.n)*16)==CUDA_SUCCESS;}
  okay=okay&&cuMemcpyDtoH(tets.data(),b.t,tets.size())==CUDA_SUCCESS&&cuMemcpyDtoH(rest.data(),b.rest,rest.size())==CUDA_SUCCESS;
  if(!okay)break;
  if(j)meta<<",";meta<<"{\"body\":"<<b.id<<",\"nodes\":"<<b.n<<",\"tets\":"<<b.nt<<",\"sections\":{";
  bool first=true;const char* fields[]={"position","velocity","delta_position","contact_delta"};
  auto append=[&](std::string key,const unsigned char* data,size_t bytes,const char* dtype,unsigned rows,unsigned width){
   if(!first)meta<<",";first=false;meta<<std::quoted(key)<<":{\"offset\":"<<offset<<",\"bytes\":"<<bytes<<",\"dtype\":"<<std::quoted(dtype)<<",\"shape\":["<<rows<<","<<width<<"]}";
   if(std::fwrite(data,1,bytes,binary)!=bytes)okay=false;offset+=bytes;
  };
  for(unsigned phase=0;phase<2;++phase){auto& data=phase?after:before;for(unsigned k=0;k<4;++k)append(std::string(phase?"after_":"before_")+fields[k],data.data()+size_t(k)*b.n*16,size_t(b.n)*16,"<f4",b.n,4);}
  append("tet_indices",tets.data(),tets.size(),"<u4",b.nt,4);append("reference_determinants",rest.data(),rest.size(),"<f8",b.nt,1);meta<<"}}";
 }
 meta<<"],\"bytes\":"<<offset<<",\"complete\":"<<(okay?"true":"false")<<"}\n";
 if(std::fclose(binary)!=0)okay=false;
 FILE* json=std::fopen((stem+".json").c_str(),"wx");if(!json)return ct_fail("snapshot metadata create");std::string text=meta.str();if(std::fwrite(text.data(),1,text.size(),json)!=text.size())okay=false;if(std::fclose(json)!=0)okay=false;
 if(!okay)return ct_fail("snapshot transfer or write");++ct_snapshots;return true;
}
bool ct_before(const char* name){
 if(!ct_enabled||!cg_enabled||std::strncmp(name,"sb_",3)||!cg_module)return true;
 if(!ct_initialize())return false;
 if(cuCtxSynchronize()!=CUDA_SUCCESS)return ct_fail("before synchronization");
 for(unsigned j=0;j<2;++j)if(!ct_copy_current(ct_body[j].before,j))return ct_fail("before checkpoint");
 ct_name=name;ct_pending=true;return true;
}
bool ct_after(const char* name,CUstream stream,Launch original){
 if(!ct_enabled||!ct_pending)return true;
 if(ct_name!=name)return ct_fail("stage pairing");
 unsigned invalid_mask=0;
 for(unsigned j=0;j<2;++j){auto& b=cg_body[j];auto& c=ct_body[j];
  if(cuMemsetD8(c.result,255,8)!=CUDA_SUCCESS||cuMemsetD8(c.result+8,0,8)!=CUDA_SUCCESS)return ct_fail("check reset");
  void* args[]={&b.p,&b.t,&b.rest,&b.nt,&c.result};
  if(original(ct_check,(b.nt+127)/128,1,1,128,1,1,0,stream,args,nullptr)!=CUDA_SUCCESS||cuCtxSynchronize()!=CUDA_SUCCESS)return ct_fail("check launch");
  unsigned long long values[2];if(cuMemcpyDtoH(values,c.result,16)!=CUDA_SUCCESS)return ct_fail("check read");if(values[1]>0)invalid_mask|=1u<<j;
 }
 ++ct_observations;
 if(invalid_mask&~ct_invalid_mask){
  ++ct_edges;ct_edge_name=name;ct_edge_step=ss_steps;ct_edge_call=cg_calls;
  for(unsigned j=0;j<2;++j){auto& c=ct_body[j];if(cuMemcpyDtoD(c.edge_before,c.before,size_t(cg_body[j].n)*64)!=CUDA_SUCCESS||!ct_copy_current(c.edge_after,j))return ct_fail("edge checkpoint");}
  if(ct_edges==1&&!ct_dump("first-invalid",true))return false;
 }
 ct_invalid_mask=invalid_mask;ct_pending=false;return true;
}
bool ct_rejected(){
 if(!ct_enabled||!ct_ready||ct_rejection_saved)return true;ct_rejection_saved=true;
 if(cuCtxSynchronize()!=CUDA_SUCCESS)return ct_fail("rejection synchronization");
 if(ct_edges&&!ct_dump("latest-invalid",true))return false;
 return ct_dump("guard-rejection",false);
}
