// Restricted two-body mapping bridge. Frame-boundary validation, no per-cycle readback.
struct BPDesc {CUdeviceptr p=0,v=0,d=0,cv=0,ends=0,saved=0,scratch=0;unsigned n=0,nt=0;};
static_assert(sizeof(BPDesc)==64,"batch descriptor ABI");
struct BPBody {BPDesc d;unsigned id=0;std::vector<unsigned> ends;std::vector<float> mass;CUdeviceptr checkpoint=0;};
BPBody bp_body[2];bool bp_enabled=false,bp_open=false,bp_average=false,bp_inframe=false;int bp_mode=0;
std::string bp_ptx;CUcontext bp_context=nullptr;CUmodule bp_module=nullptr;CUfunction bp_fn[4]{};
CUdeviceptr bp_table=0,bp_descriptors=0,bp_stats=0,bp_cpids=0;CUstream bp_stream=nullptr;
unsigned bp_parts=0,bp_next=0,bp_maxnodes=0;float bp_invdt=0;
unsigned long long bp_frames=0,bp_cycles=0,bp_partitions=0,bp_published=0,bp_errors=0,bp_syncs=0,bp_rollbacks=0;
bool bp_fail(const char* why){++bp_errors;++errors;std::fprintf(stderr,"Batched CP rejected: %s\n",why);return false;}
template<class T>T bp_field(const unsigned char* p,size_t o){T v;std::memcpy(&v,p+o,sizeof(v));return v;}
bool bp_range(CUdeviceptr p,size_t bytes){CUdeviceptr base=0;size_t size=0;CUcontext owner=nullptr;return p&&cuPointerGetAttribute(&owner,CU_POINTER_ATTRIBUTE_CONTEXT,p)==CUDA_SUCCESS&&owner==bp_context&&cuMemGetAddressRange(&base,&size,p)==CUDA_SUCCESS&&p>=base&&p-base<=size&&bytes<=size-size_t(p-base);}
bool bp_read(void* host,CUdeviceptr p,size_t bytes){return bp_range(p,bytes)&&cuMemcpyDtoH(host,p,bytes)==CUDA_SUCCESS;}
bool bp_restore(){
 bool ok=cuCtxSynchronize()==CUDA_SUCCESS;
 for(auto& b:bp_body){if(!b.checkpoint)continue;size_t bytes=size_t(b.d.n)*16;
  for(unsigned k=0;k<3;++k)if(cuMemcpyDtoD(k==0?b.d.p:k==1?b.d.v:b.d.d,b.checkpoint+k*bytes,bytes)!=CUDA_SUCCESS)ok=false;
 }++bp_rollbacks;return ok;
}
bool bp_init(CUcontext current){
 bp_context=current;
 if(cuModuleLoad(&bp_module,bp_ptx.c_str())!=CUDA_SUCCESS)return bp_fail("module load");
 const char* names[]={"batch_mm_split","batch_mm_prepare","batch_mm_merge","batch_mm_validate"};
 for(unsigned k=0;k<4;++k)if(cuModuleGetFunction(&bp_fn[k],bp_module,names[k])!=CUDA_SUCCESS)return bp_fail("helper symbol");
 if(cuMemAlloc(&bp_descriptors,2*sizeof(BPDesc))!=CUDA_SUCCESS||cuMemAlloc(&bp_stats,8*8)!=CUDA_SUCCESS||cuMemsetD8(bp_stats,0,8*8)!=CUDA_SUCCESS)return bp_fail("descriptor/counter allocation");return true;
}
bool bp_bind(void** params){
 auto table=*static_cast<CUdeviceptr*>(params[0]);auto ids_ptr=*static_cast<CUdeviceptr*>(params[1]);unsigned ids[2];
 if(bp_table&&bp_table!=table)return bp_fail("table changed");
 if(!bp_read(ids,ids_ptr,8)||ids[0]==ids[1]||ids[0]>=4096||ids[1]>=4096)return bp_fail("body indices");
 BPDesc descriptors[2];
 for(unsigned j=0;j<2;++j){auto& b=bp_body[j];unsigned char m[432];
  if(!bp_read(m,table+size_t(ids[j])*432,432))return bp_fail("metadata range");
  BPDesc d;d.p=bp_field<CUdeviceptr>(m,96);d.v=bp_field<CUdeviceptr>(m,88);d.d=bp_field<CUdeviceptr>(m,128);d.cv=bp_field<CUdeviceptr>(m,176);d.ends=bp_field<CUdeviceptr>(m,200);d.n=bp_field<unsigned>(m,372);d.nt=bp_field<unsigned>(m,376);unsigned parts=bp_field<unsigned>(m,380);
  if(!d.n||d.n>100000||!d.nt||d.nt>1000000||bp_field<unsigned>(m,412)!=1||!parts||parts>8||(bp_parts&&parts!=bp_parts))return bp_fail("unsupported body layout");
  bp_parts=parts;std::vector<unsigned> ends(d.n);
  if(!bp_read(ends.data(),d.ends,size_t(d.n)*4))return bp_fail("copy chains");
  unsigned previous=0;for(auto end:ends){if(end<=previous||end>4*d.nt)return bp_fail("empty copy chain");previous=end;}
  if(!bp_range(d.cv,size_t(4*d.nt+previous)*16)||!bp_range(d.d,size_t(d.n)*16))return bp_fail("copy/delta range");
  std::vector<float> p(size_t(d.n)*4),v(size_t(d.n)*4);
  if(!bp_read(p.data(),d.p,p.size()*4)||!bp_read(v.data(),d.v,v.size()*4))return bp_fail("physical state range");
  for(unsigned i=0;i<d.n;++i){for(unsigned k=0;k<4;++k)if(!std::isfinite(p[4*i+k])||!std::isfinite(v[4*i+k]))return bp_fail("nonfinite frame state");if(p[4*i+3]<=0||p[4*i+3]!=v[4*i+3])return bp_fail("invalid physical mass");}
  if(b.d.n){
   if(ids[j]!=b.id||d.p!=b.d.p||d.v!=b.d.v||d.d!=b.d.d||d.cv!=b.d.cv||d.ends!=b.d.ends||d.n!=b.d.n||d.nt!=b.d.nt||ends!=b.ends)return bp_fail("body identity changed");
   for(unsigned i=0;i<d.n;++i)if(p[4*i+3]!=b.mass[i])return bp_fail("physical mass changed");
  }else{
   b.id=ids[j];b.d=d;b.ends=ends;b.mass.resize(d.n);for(unsigned i=0;i<d.n;++i)b.mass[i]=p[4*i+3];
   if(cuMemAlloc(&b.d.saved,size_t(d.n)*16)!=CUDA_SUCCESS||cuMemAlloc(&b.d.scratch,size_t(d.n)*48)!=CUDA_SUCCESS||cuMemAlloc(&b.checkpoint,size_t(d.n)*48)!=CUDA_SUCCESS)return bp_fail("body scratch allocation");
  }
  descriptors[j]=b.d;bp_maxnodes=std::max(bp_maxnodes,d.n);
 }
 for(unsigned a=0;a<2;++a)for(unsigned b=a+1;b<2;++b){auto& x=bp_body[a].d;auto& y=bp_body[b].d;if(x.p<y.p+size_t(y.n)*16&&y.p<x.p+size_t(x.n)*16)return bp_fail("overlapping bodies");}
 if(cuMemcpyHtoD(bp_descriptors,descriptors,sizeof(descriptors))!=CUDA_SUCCESS)return bp_fail("descriptor upload");bp_table=table;return true;
}
bool bp_launch(unsigned phase,CUstream stream,Launch original){
 unsigned bodies=2;void* args[]={&bp_descriptors,&bodies,&bp_invdt,&bp_stats,&bp_mode};void* validate[]={&bp_descriptors,&bodies,&bp_stats};
 return original(bp_fn[phase],(bp_maxnodes+127)/128,2,1,128,1,1,0,stream,phase==3?validate:args,nullptr)==CUDA_SUCCESS;
}
bool bp_before(const char* name,void** params,void** extra,unsigned gy,unsigned gz,CUstream stream,Launch original){
 if(!bp_enabled)return true;if(bp_errors)return false;
 bool pre=!std::strcmp(name,"sb_gm_preIntegrateLaunch"),part=!std::strcmp(name,"sb_gm_cp_solveTetrahedronsPartitionLaunch"),avg=!std::strcmp(name,"sb_gm_cp_averageVertsLaunch"),final=!std::strcmp(name,"sb_gm_finalizeVelocitiesLaunch");
 if(!pre&&!part&&!avg&&!final){if(bp_open&&!std::strncmp(name,"sb_",3))return bp_fail("unexpected stage in CP cycle");return true;}
 if(!params||!params[0]||!params[1]||extra||gy!=2||gz!=1)return bp_fail("launch ABI");
 CUcontext context=nullptr;CUstreamCaptureStatus capture;
 if(cuCtxGetCurrent(&context)!=CUDA_SUCCESS||!context||(bp_context&&context!=bp_context)||cuStreamIsCapturing(stream,&capture)!=CUDA_SUCCESS||capture!=CU_STREAM_CAPTURE_STATUS_NONE)return bp_fail("context/capture");
 if(pre){
  if(bp_open||bp_inframe)return bp_fail("unfinished frame");
  if(!bp_context&&!bp_init(context))return false;
  if(cuCtxSynchronize()!=CUDA_SUCCESS)return bp_fail("frame synchronization");++bp_syncs;
  unsigned long long stats[8];if(!bp_read(stats,bp_stats,sizeof(stats))||stats[6])return bp_fail("device validation");
  if(!bp_bind(params))return false;bp_cpids=0;bp_inframe=true;++bp_frames;return true;
 }
 if(!bp_inframe||*static_cast<CUdeviceptr*>(params[0])!=bp_table)return bp_fail("unbound frame");
 if(final){if(bp_open)return bp_fail("unfinished final cycle");bp_inframe=false;return true;}
 if(!params[2])return bp_fail("missing CP arguments");
 CUdeviceptr ids_ptr=*static_cast<CUdeviceptr*>(params[1]);
 if(!bp_cpids){unsigned ids[2];if(!bp_read(ids,ids_ptr,8)||ids[0]!=bp_body[0].id||ids[1]!=bp_body[1].id)return bp_fail("CP body order");bp_cpids=ids_ptr;}else if(ids_ptr!=bp_cpids)return bp_fail("CP body-list pointer changed");
 if(part){
  if(!params[3]||!params[4]||!params[5]||*static_cast<unsigned*>(params[2])!=2||*static_cast<unsigned char*>(params[5])!=1)return bp_fail("CP material mode");
  unsigned index=*static_cast<unsigned*>(params[4]);float invdt=*static_cast<float*>(params[3]);
  if(!std::isfinite(invdt)||invdt<=0||index!=bp_next||index>=bp_parts||bp_average)return bp_fail("partition sequence/timestep");
  if(index==0){
   if(bp_open)return bp_fail("duplicate cycle");bp_stream=stream;bp_invdt=invdt;
   for(auto& b:bp_body){size_t bytes=size_t(b.d.n)*16;for(unsigned k=0;k<3;++k)if(cuMemcpyDtoDAsync(b.checkpoint+k*bytes,k==0?b.d.p:k==1?b.d.v:b.d.d,bytes,stream)!=CUDA_SUCCESS)return bp_fail("checkpoint copy");}
   bp_open=true;++bp_cycles;if(!bp_launch(0,stream,original))return bp_fail("split launch");
  }else if(!bp_open||stream!=bp_stream||invdt!=bp_invdt)return bp_fail("cycle stream/timestep changed");
  ++bp_next;
 }else{
  if(!bp_open||bp_average||bp_next!=bp_parts||stream!=bp_stream||*static_cast<float*>(params[2])!=bp_invdt)return bp_fail("merge sequence");
  if(!bp_launch(1,stream,original))return bp_fail("prepare launch");bp_average=true;
 }return true;
}
bool bp_after(const char* name,CUstream stream,Launch original){
 if(!bp_enabled)return true;
 if(!std::strcmp(name,"sb_gm_cp_solveTetrahedronsPartitionLaunch")){++bp_partitions;return true;}
 if(std::strcmp(name,"sb_gm_cp_averageVertsLaunch"))return true;
 if(!bp_open||!bp_average||!bp_launch(2,stream,original)||!bp_launch(3,stream,original))return bp_fail("merge/validation launch");
 ++bp_published;bp_open=bp_average=false;bp_next=0;return true;
}
