// Additional read-only capture of exact contact records and native call scalars.
static FILE* cx_file=nullptr;static unsigned long long cx_bytes=0;
static CUdeviceptr cx_rows=0,cx_count_ptr=0;static unsigned cx_count=0;
static bool cx_contact=false;static std::string cx_parameters="{}";
static bool cx_call(const char* name,void** params){
 cx_contact=!std::strcmp(name,"cloth_solveClothClothDeltaVTLaunch")||!std::strcmp(name,"cloth_solveClothClothDeltaEELaunch");
 cx_parameters="{}";cx_rows=cx_count_ptr=0;cx_count=0;
 if(cx_contact){
  if(!params||!params[0]||!params[1]||!params[2]||!params[3]||*static_cast<CUdeviceptr*>(params[0])!=cs_table)return cs_fail();
  cx_rows=*static_cast<CUdeviceptr*>(params[1]);cx_count_ptr=*static_cast<CUdeviceptr*>(params[2]);
  if(cuCtxSynchronize()!=CUDA_SUCCESS||!cs_read(&cx_count,cx_count_ptr,4)||cx_count>100000)return cs_fail();
  float value=*static_cast<float*>(params[3]);if(!std::isfinite(value))return cs_fail();
  char text[128];std::snprintf(text,sizeof(text),"{\"param_3_f32\":%.9g,\"contact_count\":%u}",value,cx_count);cx_parameters=text;
 }else if(!std::strcmp(name,"cloth_finalizeVelocitiesLaunch")){
  if(!params||!params[2]||!params[3]||!params[4])return cs_fail();
  float a=*static_cast<float*>(params[2]),b=*static_cast<float*>(params[3]);unsigned flag=*static_cast<unsigned char*>(params[4]);
  if(!std::isfinite(a)||!std::isfinite(b))return cs_fail();
  char text[160];std::snprintf(text,sizeof(text),"{\"param_2_f32\":%.9g,\"param_3_f32\":%.9g,\"param_4_u8\":%u}",a,b,flag);cx_parameters=text;
 }else if(!std::strcmp(name,"cloth_averageTrianglePairVertsLaunch")){
  if(!params||!params[2]||!params[3])return cs_fail();
  float a=*static_cast<float*>(params[2]);unsigned flag=*static_cast<unsigned char*>(params[3]);if(!std::isfinite(a))return cs_fail();
  char text[128];std::snprintf(text,sizeof(text),"{\"param_2_f32\":%.9g,\"param_3_u8\":%u}",a,flag);cx_parameters=text;
 }
 return true;
}
static bool cx_section(const char* name,const void* data,size_t bytes,bool comma){
 if(!cx_file||cx_bytes+bytes>1024ull*1024*1024)return cs_fail();
 if(bytes&&std::fwrite(data,1,bytes,cx_file)!=bytes)return cs_fail();
 if(std::fprintf(cs_index,"%s\"%s\":[%llu,%zu]",comma?",":"",name,cx_bytes,bytes)<0)return cs_fail();cx_bytes+=bytes;return true;
}
static bool cx_device_section(const char* name,CUdeviceptr pointer,size_t bytes){
 if(bytes>1024ull*1024*1024)return cs_fail();std::vector<unsigned char> data(bytes);
 if(bytes&&!cs_read(data.data(),pointer,bytes))return false;return cx_section(name,data.data(),bytes,true);
}
static bool cx_snapshot(){
 unsigned char descriptor[400];if(!cs_read(descriptor,cs_table+400ull*cs_id,400))return false;
 if(std::fprintf(cs_index,",\"body_index\":%u,\"native_parameters\":%s,\"extra_sections\":{",cs_id,cx_parameters.c_str())<0)return cs_fail();
 if(!cx_section("body_descriptor",descriptor,400,false))return false;
 if(cx_contact){
  unsigned count=0;if(!cs_read(&count,cx_count_ptr,4)||count!=cx_count)return cs_fail();
  unsigned triangles=cs_get<unsigned>(descriptor,320);if(triangles<1||triangles>200000)return cs_fail();
  if(!cx_device_section("contact_rows",cx_rows,size_t(count)*16))return false;
  if(!cx_device_section("triangles",cs_get<CUdeviceptr>(descriptor,72),size_t(triangles)*16))return false;
  if(!cx_device_section("previous_contact_positions",cs_get<CUdeviceptr>(descriptor,32),size_t(cs_n)*16))return false;
 }
 return std::fprintf(cs_index,"}")<0?cs_fail():true;
}
