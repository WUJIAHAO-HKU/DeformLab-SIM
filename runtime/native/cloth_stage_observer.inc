// Bounded read-only snapshots; must be included inside the census namespace.
static bool cs_enabled=false,cs_pair=false;
static FILE* cs_bin=nullptr;static FILE* cs_index=nullptr;
static unsigned cs_first=0,cs_last=0,cs_step=0,cs_n=0,cs_errors=0;
static unsigned long long cs_records=0,cs_pairs=0,cs_bytes=0;
static CUcontext cs_context=nullptr;static CUdeviceptr cs_table=0,cs_ptrs[5]{};
static unsigned cs_id=0;static std::string cs_name;
static const char* cs_fields[5]={"position_inverse_mass","velocity_inverse_mass","pending_contact_delta","accumulated_position_delta","accumulated_velocity_delta"};
static bool cs_fail(){++cs_errors;return false;}
static bool cs_read(void* dst,CUdeviceptr src,size_t bytes){
 CUdeviceptr base=0;size_t size=0;
 if(!src||!bytes||cuMemGetAddressRange(&base,&size,src)!=CUDA_SUCCESS||src<base||src-base>size||bytes>size-(src-base))return cs_fail();
 return cuMemcpyDtoH(dst,src,bytes)==CUDA_SUCCESS?true:cs_fail();
}
template<class T> static T cs_get(const unsigned char* src,size_t off){T value;std::memcpy(&value,src+off,sizeof(value));return value;}
static bool cs_bind(void** params,unsigned gy){
 if(!params||!params[0]||!params[1]||gy!=1)return cs_fail();
 CUcontext ctx=nullptr;if(cuCtxGetCurrent(&ctx)!=CUDA_SUCCESS||!ctx)return cs_fail();
 if(cuCtxSynchronize()!=CUDA_SUCCESS)return cs_fail();
 CUdeviceptr table=*static_cast<CUdeviceptr*>(params[0]),ids=*static_cast<CUdeviceptr*>(params[1]);unsigned id=0;
 if(!cs_read(&id,ids,4)||id>65535)return cs_fail();
 unsigned char desc[400];if(!cs_read(desc,table+400ull*id,sizeof(desc)))return false;
 unsigned n=cs_get<unsigned>(desc,316);if(n<3||n>100000)return cs_fail();
 CUdeviceptr ptrs[5];unsigned offsets[5]={16,8,248,256,264};
 for(unsigned j=0;j<5;++j){ptrs[j]=cs_get<CUdeviceptr>(desc,offsets[j]);if(!ptrs[j])return cs_fail();}
 if(cs_context){
  if(ctx!=cs_context||table!=cs_table||id!=cs_id||n!=cs_n)return cs_fail();
  for(unsigned j=0;j<5;++j)if(ptrs[j]!=cs_ptrs[j])return cs_fail();
 }else{cs_context=ctx;cs_table=table;cs_id=id;cs_n=n;std::copy(ptrs,ptrs+5,cs_ptrs);}
 return true;
}
static bool cs_selected(const char* name){
 static const char* names[]={"cloth_preIntegrateLaunch","cloth_solveTrianglePairEnergyLaunch","cloth_averageTrianglePairVertsLaunch",
  "cloth_accumulateInPlaneDampingDeltaVelocity","cloth_accumulateBendingDampingDeltaVelocity","cloth_applyAccumulatedDeltaVelocity",
  "cloth_applyExternalDeltasLaunch","cloth_solveRigidClothCollisionTGSLaunch","cloth_solveRigidClothAttachmentTGSLaunch",
  "cloth_solveClothClothDeltaVTLaunch","cloth_solveClothClothDeltaEELaunch","cloth_stepLaunch","cloth_finalizeVelocitiesLaunch"};
 for(const char* value:names)if(!std::strcmp(value,name))return true;return false;
}
static bool cs_snapshot(const char* phase){
 CUcontext ctx=nullptr;if(cuCtxGetCurrent(&ctx)!=CUDA_SUCCESS||ctx!=cs_context||cuCtxSynchronize()!=CUDA_SUCCESS)return cs_fail();
 const size_t bytes=size_t(cs_n)*16;
 if(cs_bytes+5*bytes>1024ull*1024*1024)return cs_fail();
 std::vector<unsigned char> data(bytes);
 if(std::fprintf(cs_index,"{\"step\":%u,\"pair\":%llu,\"kernel\":\"%s\",\"phase\":\"%s\",\"nodes\":%u,\"sections\":{",cs_step,cs_pairs,cs_name.c_str(),phase,cs_n)<0)return cs_fail();
 for(unsigned j=0;j<5;++j){
  if(!cs_read(data.data(),cs_ptrs[j],bytes))return false;
  if(std::fwrite(data.data(),1,bytes,cs_bin)!=bytes)return cs_fail();
  if(std::fprintf(cs_index,"%s\"%s\":[%llu,%zu]",j?",":"",cs_fields[j],cs_bytes,bytes)<0)return cs_fail();
  cs_bytes+=bytes;
 }
 if(std::fprintf(cs_index,"}}\n")<0)return cs_fail();++cs_records;return true;
}
static bool cs_before(const char* name,void** params,unsigned gy){
 if(!cs_enabled)return true;
 if(!std::strcmp(name,"cloth_preIntegrateLaunch")){
  if(!cs_bind(params,gy))return false;++cs_step;
 }
 if(cs_step<cs_first||cs_step>cs_last||!cs_selected(name))return true;
 if(!cs_context||cs_pair)return cs_fail();
 cs_name=name;if(!cs_snapshot("before"))return false;cs_pair=true;return true;
}
static bool cs_after(const char* name){
 if(!cs_enabled||!cs_pair)return true;
 if(cs_name!=name)return cs_fail();
 if(!cs_snapshot("after"))return false;cs_pair=false;++cs_pairs;return true;
}
