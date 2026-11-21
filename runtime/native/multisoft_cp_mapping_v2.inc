// Two-body restricted CP mapping trial; never substitutes a material law.
bool mm_enabled=false,mm_open=false,mm_average=false;int mm_mode=0;
unsigned mm_next=0,mm_parts=0;float mm_invdt=0;
unsigned long long mm_cycles=0,mm_partitions=0,mm_published=0,mm_errors=0;
FILE* mm_file=nullptr;CUfunction mm_fn[3]{};
struct MMBody{
 CUdeviceptr cv=0,ends=0,saved=0,scratch=0,checkpoint=0;
 unsigned total=0,minimum=0,maximum=0;
 std::vector<unsigned> counts;std::vector<float> p,v,mass;
 double before[7]{},native[7]{};
};MMBody mm_body[2];
bool mm_fail(const char* reason){++mm_errors;++errors;std::fprintf(stderr,"CP mapping rejected: %s\n",reason);return false;}
bool mm_totals(unsigned j,double* out){
 auto& b=mm_body[j];auto& g=cg_body[j];b.p.resize(size_t(g.n)*4);b.v.resize(size_t(g.n)*4);
 if(!ss_read(b.p.data(),g.p,b.p.size()*4)||!ss_read(b.v.data(),g.v,b.v.size()*4))return false;
 std::fill(out,out+7,0.);
 for(unsigned i=0;i<g.n;++i){
  if(b.p[4*i+3]!=b.mass[i]||b.v[4*i+3]!=b.mass[i])return false;
  double p[3],v[3],m=1./b.mass[i];
  for(unsigned k=0;k<3;++k){p[k]=b.p[4*i+k];v[k]=b.v[4*i+k];if(!std::isfinite(p[k])||!std::isfinite(v[k]))return false;out[k]+=m*v[k];out[6]+=.5*m*v[k]*v[k];}
  out[3]+=m*(p[1]*v[2]-p[2]*v[1]);out[4]+=m*(p[2]*v[0]-p[0]*v[2]);out[5]+=m*(p[0]*v[1]-p[1]*v[0]);
 }return true;
}
bool mm_restore(){
 bool okay=cuCtxSynchronize()==CUDA_SUCCESS;
 for(unsigned j=0;j<2;++j){auto& b=mm_body[j];auto& g=cg_body[j];
  if(!b.checkpoint)continue;size_t bytes=size_t(g.n)*16;
  for(unsigned k=0;k<3;++k){CUdeviceptr target=k==0?g.p:k==1?g.v:g.delta;if(cuMemcpyDtoD(target,b.checkpoint+k*bytes,bytes)!=CUDA_SUCCESS)okay=false;}
 }return okay;
}
bool mm_bind(unsigned j){
 auto& b=mm_body[j];auto& g=cg_body[j];unsigned char m[432];
 if(!g.n||!ss_read(m,ss_table+size_t(g.id)*432,432))return mm_fail("body metadata");
 auto cv=ss_field<CUdeviceptr>(m,176),ends=ss_field<CUdeviceptr>(m,200);
 unsigned parts=ss_field<unsigned>(m,380),kind=ss_field<unsigned>(m,412);
 if(kind!=1||!parts||parts>8||(mm_parts&&mm_parts!=parts)||!ss_range(ends,size_t(g.n)*4))return mm_fail("unsupported CP layout");
 mm_parts=parts;std::vector<unsigned> counts(g.n);
 if(!ss_read(counts.data(),ends,size_t(g.n)*4))return mm_fail("chain counts");
 unsigned previous=0,lo=~0u,hi=0;
 for(auto end:counts){if(end<=previous||end>4*g.nt)return mm_fail("empty or invalid copy chain");unsigned c=end-previous;lo=std::min(lo,c);hi=std::max(hi,c);previous=end;}
 if(!ss_range(cv,size_t(4*g.nt+previous)*16))return mm_fail("copy velocity range");
 if(b.ends){if(cv!=b.cv||ends!=b.ends||counts!=b.counts)return mm_fail("CP identity changed");return true;}
 b.cv=cv;b.ends=ends;b.counts=counts;b.total=previous;b.minimum=lo;b.maximum=hi;
 b.p.resize(size_t(g.n)*4);if(!ss_read(b.p.data(),g.p,b.p.size()*4))return mm_fail("root mass read");
 b.mass.resize(g.n);for(unsigned i=0;i<g.n;++i){float w=b.p[4*i+3];if(!std::isfinite(w)||w<=0||!std::isfinite(w*hi))return mm_fail("invalid free-node mass");b.mass[i]=w;}
 if(cuMemAlloc(&b.saved,size_t(g.n)*16)!=CUDA_SUCCESS||cuMemAlloc(&b.scratch,size_t(g.n)*48)!=CUDA_SUCCESS||cuMemAlloc(&b.checkpoint,size_t(g.n)*48)!=CUDA_SUCCESS)return mm_fail("CP scratch allocation");
 return true;
}
bool mm_launch(unsigned k,unsigned j,CUstream stream,Launch original){
 auto& b=mm_body[j];auto& g=cg_body[j];
 void* split[]={&g.p,&g.v,&b.ends,&g.n,&b.saved};
 void* prepare[]={&g.p,&g.v,&g.delta,&g.n,&b.saved,&b.scratch};
 void* merge[]={&g.p,&g.v,&g.delta,&b.cv,&b.ends,&g.n,&g.nt,&b.scratch,&mm_invdt};
 return original(mm_fn[k],(g.n+127)/128,1,1,128,1,1,0,stream,k==0?split:k==1?prepare:merge,nullptr)==CUDA_SUCCESS;
}
bool mm_before(const char* name,void** params,unsigned gy,CUstream stream,Launch original){
 if(!mm_enabled)return true;if(mm_errors)return false;
 bool pre=!std::strcmp(name,"sb_gm_preIntegrateLaunch"),part=!std::strcmp(name,"sb_gm_cp_solveTetrahedronsPartitionLaunch"),avg=!std::strcmp(name,"sb_gm_cp_averageVertsLaunch");
 if(pre){
  if(mm_open||!cg_module)return mm_fail("unfinished CP cycle before frame");
  if(!mm_fn[0]){const char* names[]={"mm_split","mm_prepare","mm_merge"};for(unsigned k=0;k<3;++k)if(cuModuleGetFunction(&mm_fn[k],cg_module,names[k])!=CUDA_SUCCESS)return mm_fail("CP helper symbols");}
  for(unsigned j=0;j<2;++j)if(!mm_bind(j))return false;
 }
 if(!part&&!avg){if(mm_open&&!std::strncmp(name,"sb_",3))return mm_fail("unexpected softbody stage inside CP cycle");return true;}
 if(!mm_parts||gy!=2||!params||!params[0]||!params[1]||!params[2]||*static_cast<CUdeviceptr*>(params[0])!=ss_table)return mm_fail("CP launch shape or table");
 unsigned ids[2];if(!ss_read(ids,*static_cast<CUdeviceptr*>(params[1]),8)||ids[0]!=cg_body[0].id||ids[1]!=cg_body[1].id)return mm_fail("CP body order");
 if(part){
  if(!params[3]||!params[4]||!params[5]||*static_cast<unsigned*>(params[2])!=gy||*static_cast<unsigned char*>(params[5])!=1)return mm_fail("CP constitutive mode");
  unsigned index=*static_cast<unsigned*>(params[4]);float invdt=*static_cast<float*>(params[3]);
  if(!std::isfinite(invdt)||invdt<=0||index!=mm_next||index>=mm_parts||mm_average)return mm_fail("CP partition sequence");
  if(index==0){
   if(mm_open)return mm_fail("duplicate CP begin");mm_invdt=invdt;
   if(cuCtxSynchronize()!=CUDA_SUCCESS)return mm_fail("CP begin synchronization");
   for(unsigned j=0;j<2;++j){auto& b=mm_body[j];auto& g=cg_body[j];size_t bytes=size_t(g.n)*16;
    if(!mm_totals(j,b.before))return mm_fail("CP physical masses or finite input");
    for(unsigned k=0;k<3;++k){CUdeviceptr source=k==0?g.p:k==1?g.v:g.delta;if(cuMemcpyDtoD(b.checkpoint+k*bytes,source,bytes)!=CUDA_SUCCESS)return mm_fail("CP checkpoint");}
   }
   mm_open=true;++mm_cycles;
   if(mm_mode)for(unsigned j=0;j<2;++j)if(!mm_launch(0,j,stream,original)){mm_restore();return mm_fail("CP mass split");}
  }else if(!mm_open||invdt!=mm_invdt)return mm_fail("CP timestep changed");
  ++mm_next;
 }else{
  if(!mm_open||mm_next!=mm_parts||mm_average||*static_cast<float*>(params[2])!=mm_invdt)return mm_fail("CP merge sequence");
  if(mm_mode)for(unsigned j=0;j<2;++j)if(!mm_launch(1,j,stream,original)){mm_restore();return mm_fail("CP prepare");}
  mm_average=true;
 }return true;
}
bool mm_after(const char* name,CUstream stream,Launch original){
 if(!mm_enabled)return true;
 if(!std::strcmp(name,"sb_gm_cp_solveTetrahedronsPartitionLaunch")){++mm_partitions;return true;}
 if(std::strcmp(name,"sb_gm_cp_averageVertsLaunch"))return true;
 if(!mm_open||!mm_average||cuCtxSynchronize()!=CUDA_SUCCESS)return mm_fail("CP native merge synchronization");
 for(unsigned j=0;j<2;++j)if(!mm_totals(j,mm_body[j].native)){mm_restore();return mm_fail("native merge state");}
 if(mm_mode)for(unsigned j=0;j<2;++j)if(!mm_launch(2,j,stream,original)){mm_restore();return mm_fail("copy velocity merge");}
 if(cuCtxSynchronize()!=CUDA_SUCCESS){mm_restore();return mm_fail("CP publication synchronization");}
 double after[2][7];for(unsigned j=0;j<2;++j)if(!mm_totals(j,after[j])){mm_restore();return mm_fail("CP publication finite/mass check");}
 ++mm_published;
 for(unsigned j=0;j<2;++j){auto& b=mm_body[j];auto& g=cg_body[j];
  std::fprintf(mm_file,"%llu\t%llu\t%u\t%d\t%u\t%u\t%u\t%u\t%u\t%.17g",ss_steps,mm_cycles,g.id,mm_mode,g.n,b.minimum,b.maximum,b.total,mm_parts,double(mm_invdt));
  for(auto values:{b.before,b.native,after[j]})for(unsigned k=0;k<7;++k)std::fprintf(mm_file,"\t%.17g",values[k]);std::fprintf(mm_file,"\n");
 }
 if(std::ferror(mm_file))return mm_fail("CP ledger write");
 mm_open=mm_average=false;mm_next=0;return true;
}
