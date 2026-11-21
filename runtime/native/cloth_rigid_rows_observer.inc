// Read-only rigid-cloth contact records for the pinned 14-argument TGS entry.
static FILE* cr_file=nullptr;static unsigned long long cr_bytes=0;
static bool cr_active=false;static unsigned cr_count=0;
static CUdeviceptr cr_rows=0,cr_prepared=0,cr_count_pointer=0;
static float cr_h=0;
static bool cr_call(const char* name,void** params){
 cr_active=!std::strcmp(name,"cloth_solveRigidClothCollisionTGSLaunch");
 cr_count=0;cr_rows=cr_prepared=cr_count_pointer=0;cr_h=0;
 if(!cr_active)return true;
 if(!params||!params[0]||!params[1]||!params[2]||!params[3]||!params[11]||
    *static_cast<CUdeviceptr*>(params[0])!=cs_table)return cs_fail();
 cr_rows=*static_cast<CUdeviceptr*>(params[1]);cr_prepared=*static_cast<CUdeviceptr*>(params[2]);
 cr_count_pointer=*static_cast<CUdeviceptr*>(params[3]);cr_h=*static_cast<float*>(params[11]);
 if(!std::isfinite(cr_h)||cr_h<=0||cuCtxSynchronize()!=CUDA_SUCCESS||!cs_read(&cr_count,cr_count_pointer,4)||cr_count>100000)return cs_fail();
 return true;
}
static bool cr_section(const char* name,const void* data,size_t bytes,bool comma){
 if(!cr_file||cr_bytes+bytes>2ull*1024*1024*1024)return cs_fail();
 if(bytes&&std::fwrite(data,1,bytes,cr_file)!=bytes)return cs_fail();
 if(std::fprintf(cs_index,"%s\"%s\":[%llu,%zu]",comma?",":"",name,cr_bytes,bytes)<0)return cs_fail();
 cr_bytes+=bytes;return true;
}
static bool cr_device_section(const char* name,CUdeviceptr pointer,size_t bytes){
 if(bytes>64ull*1024*1024)return cs_fail();std::vector<unsigned char> data(bytes);
 if(bytes&&!cs_read(data.data(),pointer,bytes))return false;
 return cr_section(name,data.data(),bytes,true);
}
static bool cr_snapshot(){
 if(!cr_active)return true;
 unsigned count=0;if(!cs_read(&count,cr_count_pointer,4)||count!=cr_count)return cs_fail();
 unsigned char desc[400];if(!cs_read(desc,cs_table+400ull*cs_id,400))return false;
 unsigned nt=cs_get<unsigned>(desc,320);if(nt<1||nt>200000)return cs_fail();
 if(std::fprintf(cs_index,",\"rigid_contact_parameters\":{\"contact_count\":%u,\"param_11_f32\":%.9g},\"rigid_sections\":{",count,cr_h)<0)return cs_fail();
 if(!cr_section("body_descriptor",desc,400,false))return false;
 if(!cr_device_section("contact_rows",cr_rows,size_t(count)*16))return false;
 if(!cr_device_section("prepared_contacts",cr_prepared,size_t((count+31)/32)*3200))return false;
 if(!cr_device_section("triangles",cs_get<CUdeviceptr>(desc,72),size_t(nt)*16))return false;
 return std::fprintf(cs_index,"}")<0?cs_fail():true;
}
