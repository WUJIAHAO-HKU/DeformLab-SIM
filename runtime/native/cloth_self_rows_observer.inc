// Read-only single-cloth VT/EE rows before/after native validity queries and solves.
static FILE* ct_file=nullptr;static unsigned long long ct_bytes=0;
static bool ct_active=false,ct_query=false;static unsigned ct_count=0;
static CUdeviceptr ct_rows=0,ct_count_pointer=0;static float ct_h=0;
static bool ct_call(const char* name,void** params){
 ct_query=!std::strcmp(name,"cloth_queryClothClothContactVTCountLaunch")||!std::strcmp(name,"cloth_queryClothClothContactEECountLaunch");
 ct_active=ct_query||!std::strcmp(name,"cloth_solveClothClothDeltaVTLaunch")||!std::strcmp(name,"cloth_solveClothClothDeltaEELaunch");
 ct_count=0;ct_h=0;ct_rows=ct_count_pointer=0;
 if(!ct_active)return true;
 if(!params||!params[0]||!params[1]||!params[2]||*static_cast<CUdeviceptr*>(params[0])!=cs_table||cs_id!=0)return cs_fail();
 if(!ct_query){if(!params[3])return cs_fail();ct_h=*static_cast<float*>(params[3]);if(!std::isfinite(ct_h)||ct_h<=0)return cs_fail();}
 ct_rows=*static_cast<CUdeviceptr*>(params[1]);ct_count_pointer=*static_cast<CUdeviceptr*>(params[2]);
 if(cuCtxSynchronize()!=CUDA_SUCCESS||!cs_read(&ct_count,ct_count_pointer,4)||ct_count>100000)return cs_fail();
 return true;
}
static bool ct_section(const char* name,const void* data,size_t bytes,bool comma){
 if(!ct_file||ct_bytes+bytes>8ull*1024*1024*1024)return cs_fail();
 if(bytes&&std::fwrite(data,1,bytes,ct_file)!=bytes)return cs_fail();
 if(std::fprintf(cs_index,"%s\"%s\":[%llu,%zu]",comma?",":"",name,ct_bytes,bytes)<0)return cs_fail();
 ct_bytes+=bytes;return true;
}
static bool ct_device_section(const char* name,CUdeviceptr pointer,size_t bytes){
 if(bytes>64ull*1024*1024)return cs_fail();std::vector<unsigned char> data(bytes);
 if(bytes&&!cs_read(data.data(),pointer,bytes))return false;
 return ct_section(name,data.data(),bytes,true);
}
static bool ct_snapshot(){
 if(!ct_active)return true;
 unsigned count=0;if(!cs_read(&count,ct_count_pointer,4)||count!=ct_count)return cs_fail();
 unsigned char desc[400];if(!cs_read(desc,cs_table,400))return false;
 unsigned nt=cs_get<unsigned>(desc,320);if(nt<1||nt>200000)return cs_fail();
 if(std::fprintf(cs_index,",\"self_parameters\":{\"contact_count\":%u,\"query\":%s,\"h\":%.9g},\"self_sections\":{",count,ct_query?"true":"false",ct_h)<0)return cs_fail();
 if(!ct_section("body_descriptor",desc,400,false)||!ct_device_section("contact_rows",ct_rows,size_t(count)*16)||
    !ct_device_section("triangles",cs_get<CUdeviceptr>(desc,72),size_t(nt)*16)||
    !ct_device_section("previous_contact_positions",cs_get<CUdeviceptr>(desc,32),size_t(cs_n)*16)||
    !ct_device_section("dynamic_friction",cs_get<CUdeviceptr>(desc,48),size_t(cs_n)*4))return false;
 return std::fprintf(cs_index,"}")<0?cs_fail():true;
}
