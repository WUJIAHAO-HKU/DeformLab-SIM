// Read-only box/cloth vertex generator inputs and outputs for the pinned ABI.
static bool cg_active=false;
static unsigned cg_pairs=0,cg_filters=0;
static CUdeviceptr cg_ptrs[8]{};
static unsigned char cg_output[80]{};
static bool cg_call(const char* name,void** params){
 cg_active=!std::strcmp(name,"cloth_boxVertexContactGenLaunch");
 if(!cg_active)return true;
 if(!params)return cs_fail();for(unsigned i=0;i<11;++i)if(!params[i])return cs_fail();
 cg_pairs=*static_cast<unsigned*>(params[0]);cg_filters=*static_cast<unsigned*>(params[9]);
 if(cg_pairs>64||cg_filters>100000)return cs_fail();
 for(unsigned i=0;i<8;++i)cg_ptrs[i]=*static_cast<CUdeviceptr*>(params[i+1]);
 if(cg_ptrs[5]!=cs_table||cs_id!=0)return cs_fail();
 std::memcpy(cg_output,params[10],80);return true;
}
static bool cg_snapshot(){
 if(!cg_active)return true;
 std::vector<unsigned> pairs(size_t(cg_pairs)*4);
 if(cg_pairs&&!cs_read(pairs.data(),cg_ptrs[0],pairs.size()*4))return false;
 unsigned shapes=0,transforms=0;
 for(unsigned i=0;i<cg_pairs;++i){
  for(unsigned j=0;j<4;++j)if(pairs[4*i+j]>=65536)return cs_fail();
  shapes=std::max(shapes,std::max(pairs[4*i],pairs[4*i+1])+1);
  transforms=std::max(transforms,std::max(pairs[4*i+2],pairs[4*i+3])+1);
 }
 if(shapes>65536||transforms>65536)return cs_fail();
 unsigned count=0,capacity=cs_get<unsigned>(cg_output,72);
 if(!cs_read(&count,cs_get<CUdeviceptr>(cg_output,32),4)||count>capacity||count>100000)return cs_fail();
 if(std::fprintf(cs_index,",\"generation_parameters\":{\"pair_count\":%u,\"filter_count\":%u,\"shape_extent\":%u,\"transform_extent\":%u,\"output_count\":%u,\"output_capacity\":%u},\"generation_sections\":{",cg_pairs,cg_filters,shapes,transforms,count,capacity)<0)return cs_fail();
 if(!cr_section("output_descriptor",cg_output,80,false)||!cr_section("pair_records",pairs.data(),pairs.size()*4,true))return false;
 if(!cr_device_section("shape_records",cg_ptrs[4],size_t(shapes)*48)||
    !cr_device_section("transforms",cg_ptrs[1],size_t(transforms)*32)||
    !cr_device_section("contact_margins",cg_ptrs[2],size_t(transforms)*4)||
    !cr_device_section("pair_rest_distances",cg_ptrs[3],size_t(cg_pairs)*4)||
    !cr_device_section("shape_handles",cg_ptrs[6],size_t(transforms)*8)||
    !cr_device_section("filters",cg_ptrs[7],size_t(cg_filters)*32)||
    !cr_device_section("body_descriptor",cs_table,400))return false;
 const char* names[4]={"output_points","output_normals","output_barycentrics","output_rows"};
 for(unsigned i=0;i<4;++i)if(!cr_device_section(names[i],cs_get<CUdeviceptr>(cg_output,8*i),size_t(count)*16))return false;
 return std::fprintf(cs_index,"}")<0?cs_fail():true;
}
