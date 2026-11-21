// Read-only omitted-node monitoring at native kernel boundaries. Scarce-face
// frames only; deliberately intrusive and not a performance benchmark.
namespace {
bool pc_guard_bind(const char* name,void** params){
 if(pc_policy!=4||cp_frame<1||std::strcmp(name,"stepArticulation1TTGS"))return true;
 if(!params||cuCtxSynchronize()!=CUDA_SUCCESS)return pc_fail("guard articulation launch");
 unsigned char core[344];
 if(!cp_read(core,*static_cast<CUdeviceptr*>(params[0]),sizeof(core)))return pc_fail("guard articulation read");
 if(field<unsigned>(core,44)!=1||field<unsigned>(core,248)!=4||field<unsigned>(core,252)!=3)return pc_fail("guard articulation ABI");
 pc_dofs=field<CUdeviceptr>(core,184);return pc_dofs!=0||pc_fail("guard missing dofs");
}
bool pc_guard_check(const char* name,const char* phase){
 if(pc_policy!=4||!pc_guard_armed||pc_rejections)return true;
 if(cuCtxSynchronize()!=CUDA_SUCCESS||!pc_dofs)return pc_fail("guard state unavailable");
 float q=0;std::vector<float4> p(nodes);
 if(!cp_read(&q,pc_dofs+2*12160,4)||!cp_read(p.data(),positions,nodes*16)||!std::isfinite(q))return pc_fail("guard state read");
 double plane[2]={pc_planes[0],pc_planes[1]+double(q)-pc_qstart};bool valid=true;
 std::vector<float4> omitted;
 for(int side=0;side<2;++side)for(unsigned id:pc_omitted[side]){
  double g=(side==0?1.:-1.)*(double(p[id].z)-plane[side]);
  if(!std::isfinite(g)||!std::isfinite(p[id].x)||!std::isfinite(p[id].y)||std::fabs(p[id].x)>.075||std::fabs(p[id].y)>.075)valid=false;
  if(!(g>0))valid=false;omitted.push_back(p[id]);
 }
 long offset=std::ftell(pc_guard_blob);
 if(offset<0||std::fwrite(omitted.data(),16,omitted.size(),pc_guard_blob)!=omitted.size())return pc_fail("guard capture");
 std::fprintf(pc_guard_log,"{\"frame\":%d,\"seq\":%llu,\"kernel\":\"%s\",\"phase\":\"%s\",\"raw_offset\":%ld,\"nodes\":%zu,\"q_z\":%.17g,\"q_start\":%.17g,\"start_planes_m\":[%.17g,%.17g],\"planes_m\":[%.17g,%.17g],\"omitted_nodes\":[",cp_frame,pc_guard_records++,name,phase,offset,nodes,double(q),pc_qstart,pc_planes[0],pc_planes[1],plane[0],plane[1]);
 for(int side=0;side<2;++side){if(side)std::fprintf(pc_guard_log,",");std::fprintf(pc_guard_log,"[");for(unsigned j=0;j<pc_omitted[side].size();++j)std::fprintf(pc_guard_log,"%s%u",j?",":"",pc_omitted[side][j]);std::fprintf(pc_guard_log,"]");}
 std::fprintf(pc_guard_log,"],\"accepted\":%s}\n",valid?"true":"false");
 if(std::ferror(pc_guard_log))return pc_fail("guard log");
 return valid||pc_fail("omitted surface node entered contact or left fixture");
}
bool pc_guard_selected(const char* name){return !std::strncmp(name,"sb_",3)||!std::strcmp(name,"stepArticulation1TTGS");}
}
extern "C" int observer_plane_guard_frame(double qz,double band){
 if(pc_policy!=4||!std::isfinite(qz)||!std::isfinite(band)||std::fabs(band-1e-4)>1e-12)return -1;
 pc_qstart=qz;pc_band=band;return 0;
}
extern "C" int observer_plane_guard_end(double upper,double qz){
 if(pc_policy!=4||!std::isfinite(upper)||!std::isfinite(qz))return -1;
 // Independent public world pose versus internal Cartesian displacement.
 if(std::fabs(upper-(pc_planes[1]+qz-pc_qstart))>2e-6){pc_fail("guard world/joint mapping");return -2;}
 if(pc_guard_armed){float q=0;
  if(!pc_dofs||!cp_read(&q,pc_dofs+2*12160,4)||std::fabs(double(q)-qz)>2e-8){pc_fail("guard public/internal joint mismatch");return -3;}
  if(!pc_guard_check("frame-end","after"))return -4;
  std::fflush(pc_guard_log);std::fflush(pc_guard_blob);
 }
 return 0;
}
