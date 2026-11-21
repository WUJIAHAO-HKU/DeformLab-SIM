// Restricted diagnostic binding: two free-node volume bodies over z=0.12 bench.
// Composes SS common histories with native velocity stage -> guard -> native drift.
bool cg_enabled=false,cg_pending=false;int cg_mode=0;
std::string cg_ptx,cg_split_ptx;FILE* cg_file=nullptr;
CUmodule cg_module=nullptr,cg_native=nullptr;CUfunction cg_fn[9]{},cg_velocity=nullptr;
unsigned long long cg_calls=0,cg_publications=0,cg_rejections=0,cg_forced=0,cg_errors=0;
double cg_floor=.12,cg_eta=.001;
struct CGBody{
 unsigned id=0,n=0,nt=0;
 CUdeviceptr p=0,v=0,delta=0,t=0,rest=0,work=0,internal=0,floor_impulse=0;
 CUdeviceptr gradients=0,target=0,normals=0,active=0,energy=0,stats=0,incoming=0,before=0,ledger=0;
 std::vector<CUdeviceptr> allocated;
 unsigned long long host[10]{};double values[18]{};
};CGBody cg_body[2];
bool cg_fail(const char* reason){++cg_errors;++errors;std::fprintf(stderr,"Cone binding rejected: %s\n",reason);return false;}
bool cg_alloc(CGBody& b,CUdeviceptr& p,size_t bytes){
 if(cuMemAlloc(&p,bytes)!=CUDA_SUCCESS)return false;b.allocated.push_back(p);return true;
}
bool cg_bind(unsigned j){
 CGBody& b=cg_body[j];unsigned char meta[432];
 if(!ss_read(meta,ss_table+size_t(ss_bodies[j].id)*432,432))return cg_fail("metadata");
 unsigned id=ss_bodies[j].id,n=ss_field<unsigned>(meta,372),nt=ss_field<unsigned>(meta,376);
 CUdeviceptr p=ss_field<CUdeviceptr>(meta,96),v=ss_field<CUdeviceptr>(meta,88),delta=ss_field<CUdeviceptr>(meta,128),t=ss_field<CUdeviceptr>(meta,80);
 if(!nt||nt>1000000||!ss_range(p,size_t(n)*16)||!ss_range(v,size_t(n)*16)||!ss_range(delta,size_t(n)*16)||!ss_range(t,size_t(nt)*16))return cg_fail("native buffer ranges");
 if(b.n){if(b.id!=id||b.n!=n||b.nt!=nt||b.p!=p||b.v!=v||b.delta!=delta||b.t!=t)return cg_fail("body identity changed");return true;}
 b.id=id;b.n=n;b.nt=nt;b.p=p;b.v=v;b.delta=delta;b.t=t;
 std::vector<float> position(size_t(n)*4);std::vector<unsigned> tets(size_t(nt)*4);std::vector<double> determinants(nt);
 if(!ss_read(position.data(),p,position.size()*4)||!ss_read(tets.data(),t,tets.size()*4))return cg_fail("initial topology read");
 for(unsigned i=0;i<n;++i)if(!std::isfinite(position[4*i+3])||position[4*i+3]<=0)return cg_fail("fixed or invalid node unsupported");
 for(unsigned i=0;i<nt;++i){
  double e[3][3];unsigned base=tets[4*i];if(base>=n)return cg_fail("initial tet index");
  for(unsigned a=0;a<3;++a){unsigned k=tets[4*i+a+1];if(k>=n)return cg_fail("initial tet index");for(unsigned c=0;c<3;++c)e[a][c]=double(position[4*k+c])-position[4*base+c];}
  double d=e[0][0]*(e[1][1]*e[2][2]-e[1][2]*e[2][1])+e[0][1]*(e[1][2]*e[2][0]-e[1][0]*e[2][2])+e[0][2]*(e[1][0]*e[2][1]-e[1][1]*e[2][0]);
  if(!std::isfinite(d)||d<=0)return cg_fail("initial inverted or degenerate tet; positive-oriented mesh required");determinants[i]=d;
 }
 if(!cg_alloc(b,b.rest,size_t(nt)*8)||!cg_alloc(b,b.work,size_t(n)*24)||!cg_alloc(b,b.internal,size_t(n)*24)||!cg_alloc(b,b.floor_impulse,size_t(n)*24)||!cg_alloc(b,b.gradients,size_t(nt)*96)||!cg_alloc(b,b.target,size_t(nt)*8)||!cg_alloc(b,b.normals,size_t(nt)*24)||!cg_alloc(b,b.active,size_t(nt)*4)||!cg_alloc(b,b.energy,16)||!cg_alloc(b,b.stats,80)||!cg_alloc(b,b.incoming,size_t(n)*16)||!cg_alloc(b,b.before,size_t(n)*16)||!cg_alloc(b,b.ledger,144))return cg_fail("scratch allocation");
 if(cuMemcpyHtoD(b.rest,determinants.data(),size_t(nt)*8)!=CUDA_SUCCESS)return cg_fail("rest determinant upload");return true;
}
bool cg_launch(unsigned k,void** args,unsigned count,CUstream stream,Launch original){
 return original(cg_fn[k],(count+127)/128,1,1,128,1,1,0,stream,args,nullptr)==CUDA_SUCCESS;
}
bool cg_restore(){
 bool ok=cuCtxSynchronize()==CUDA_SUCCESS;
 for(auto& b:cg_body)if(b.n&&cuMemcpyDtoD(b.v,b.incoming,size_t(b.n)*16)!=CUDA_SUCCESS)ok=false;
 return ok;
}
bool cg_before(const char* name,void** params,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned shared,CUstream stream,Launch original){
 if(!cg_enabled)return true;if(cg_errors)return false;
 if(!std::strcmp(name,"sb_gm_preIntegrateLaunch")){
  if(!ss_context||gy!=2)return cg_fail("SS binding must initialize first");
  if(!cg_module){
   if(cuModuleLoad(&cg_module,cg_ptx.c_str())!=CUDA_SUCCESS||cuModuleLoad(&cg_native,cg_split_ptx.c_str())!=CUDA_SUCCESS||cuModuleGetFunction(&cg_velocity,cg_native,"owned_guard_native_velocity")!=CUDA_SUCCESS)return cg_fail("helper modules");
   const char* names[]={"cone_init","cone_prepare","cone_reset","cone_find","cone_apply","cone_verify_nodes","cone_verify_tets","cone_publish","cone_ledger"};
   for(unsigned k=0;k<9;++k)if(cuModuleGetFunction(&cg_fn[k],cg_module,names[k])!=CUDA_SUCCESS)return cg_fail("helper symbols");
  }
  for(unsigned j=0;j<2;++j)if(!cg_bind(j))return false;
 }
 if(std::strcmp(name,"sb_gm_stepSoftbodyLaunch"))return true;
 if(!cg_module||cg_pending||gy!=2||!params||!params[0]||!params[1]||!params[2]||!params[3]||!params[4]||*static_cast<CUdeviceptr*>(params[0])!=ss_table)return cg_fail("unbound native drift");
 unsigned ids[2];if(!ss_read(ids,*static_cast<CUdeviceptr*>(params[1]),8)||ids[0]!=cg_body[0].id||ids[1]!=cg_body[1].id)return cg_fail("drift body IDs");
 float hf=*static_cast<float*>(params[2]);double h=hf;unsigned char forcing=*static_cast<unsigned char*>(params[4]);
 if(!std::isfinite(h)||h<=0||forcing>1)return cg_fail("drift arguments");
 if(cuCtxSynchronize()!=CUDA_SUCCESS)return cg_fail("checkpoint synchronization");
 for(auto& b:cg_body)if(cuMemcpyDtoD(b.incoming,b.v,size_t(b.n)*16)!=CUDA_SUCCESS)return cg_fail("incoming velocity checkpoint");
 bool okay=true;
 if(forcing){okay=original(cg_velocity,gx,gy,gz,bx,by,bz,shared,stream,params,nullptr)==CUDA_SUCCESS&&cuCtxSynchronize()==CUDA_SUCCESS;if(okay)++cg_forced;}
 for(unsigned j=0;j<2&&okay;++j){
  auto& b=cg_body[j];std::fill(std::begin(b.host),std::end(b.host),0ull);std::fill(std::begin(b.values),std::end(b.values),0.);
  if(!cg_mode)continue;
  okay=cuMemcpyDtoD(b.before,b.v,size_t(b.n)*16)==CUDA_SUCCESS&&cuMemsetD8(b.stats,0,80)==CUDA_SUCCESS&&cuMemsetD8(b.ledger,0,144)==CUDA_SUCCESS;
  int plane=1;
  void* init[]={&b.p,&b.v,&b.n,&h,&cg_eta,&b.work,&b.internal,&b.floor_impulse,&b.energy,&b.stats};
  void* prepare[]={&b.p,&b.n,&b.t,&b.nt,&b.rest,&h,&cg_eta,&b.energy,&b.gradients,&b.target,&b.stats};
  void* reset[]={&b.stats};
  void* find[]={&b.t,&b.nt,&b.work,&b.gradients,&b.target,&h,&b.normals,&b.active,&b.stats};
  void* apply[]={&b.p,&b.n,&b.t,&b.nt,&b.gradients,&b.target,&b.normals,&b.active,&h,&plane,&cg_floor,&b.work,&b.internal,&b.floor_impulse,&b.stats};
  if(okay)okay=cg_launch(0,init,1,stream,original)&&cg_launch(1,prepare,b.nt,stream,original);
  for(unsigned group=0;group<16&&okay;++group){
   for(unsigned it=0;it<8&&okay;++it)okay=cg_launch(2,reset,1,stream,original)&&cg_launch(3,find,b.nt,stream,original)&&cg_launch(4,apply,1,stream,original);
   if(okay)okay=cuCtxSynchronize()==CUDA_SUCCESS&&cuMemcpyDtoH(b.host,b.stats,80)==CUDA_SUCCESS;
   if(b.host[0]||b.host[4])break;
  }
  void* verifyN[]={&b.p,&b.v,&b.n,&b.work,&h,&plane,&cg_floor,&b.energy,&b.stats};
  void* verifyT[]={&b.p,&b.t,&b.nt,&b.work,&b.gradients,&b.rest,&h,&cg_eta,&b.stats};
  void* ledger[]={&b.p,&b.incoming,&b.before,&b.work,&b.internal,&b.floor_impulse,&b.n,&b.ledger};
  if(okay)okay=cg_launch(5,verifyN,1,stream,original)&&cg_launch(6,verifyT,b.nt,stream,original)&&cg_launch(8,ledger,b.n,stream,original)&&cuCtxSynchronize()==CUDA_SUCCESS&&cuMemcpyDtoH(b.host,b.stats,80)==CUDA_SUCCESS&&cuMemcpyDtoH(b.values,b.ledger,144)==CUDA_SUCCESS;
  if(okay){
   for(double x:b.values)okay=okay&&std::isfinite(x);
   okay=okay&&!b.host[0]&&b.host[4]&&b.values[12]<=1e-12&&b.values[17]<=1e-12&&b.values[16]<=1e-10&&std::fabs(b.values[12]-b.values[13]-b.values[14])<=1e-12;
   double P=0,L=0;for(unsigned k=0;k<3;++k){P+=b.values[k]*b.values[k];L+=b.values[k+3]*b.values[k+3];}okay=okay&&P<=1e-20&&L<=1e-22;
  }
 }
 ++cg_calls;
 if(!okay){++cg_rejections;bool restored=cg_restore();std::fprintf(cg_file,"# rejected call=%llu restored=%d body0_status=%llu body1_status=%llu\n",cg_calls,int(restored),cg_body[0].host[0],cg_body[1].host[0]);std::fflush(cg_file);return cg_fail(restored?"guard or ledger certificate":"guard and rollback");}
 // Commit both bodies only after both certificates pass. Failure aborts the task;
 // rollback preserves the incoming native velocity, never a silent fallback.
 if(cg_mode)for(auto& b:cg_body){void* publish[]={&b.v,&b.n,&b.work,&b.stats};if(!cg_launch(7,publish,b.n,stream,original)){cg_restore();return cg_fail("two-body publication");}}
 if(cuCtxSynchronize()!=CUDA_SUCCESS){cg_restore();return cg_fail("publication synchronization");}
 for(auto& b:cg_body){
  if(cg_mode&&(cuMemcpyDtoH(b.host,b.stats,80)!=CUDA_SUCCESS||b.host[6]!=b.n)){cg_restore();return cg_fail("publication coverage");}
  std::fprintf(cg_file,"%llu\t%llu\t%u\t%.17g\t%u\t%d\t%llu\t%llu\t%llu",ss_steps,cg_calls,b.id,h,unsigned(forcing),cg_mode,b.host[1],b.host[2],b.host[3]);
  for(double value:b.values)std::fprintf(cg_file,"\t%.17g",value);std::fprintf(cg_file,"\n");
 }
 if(std::ferror(cg_file))return cg_fail("ledger write");cg_pending=true;return true;
}
bool cg_after(const char* name){
 if(cg_enabled&&!std::strcmp(name,"sb_gm_stepSoftbodyLaunch")){
  if(!cg_pending)return cg_fail("missing guarded step");cg_pending=false;++cg_publications;
 }return true;
}
