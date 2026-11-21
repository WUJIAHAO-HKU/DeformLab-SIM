// Experimental single-softbody / single-dynamic-rigid contact policy.
// Pinned prepare ABI, host synchronization in ALL policy arms.
namespace {
FILE* cp_log=nullptr;FILE* cp_blob=nullptr;int cp_policy=0,cp_frame=-1,cp_freeze_frame=153;
unsigned long long cp_prepares=0,cp_queries=0;CUfunction cp_countfn=nullptr;
std::mutex cp_lock;
struct CPVec {double x,y,z;CPVec operator+(CPVec b)const{return {x+b.x,y+b.y,z+b.z};}CPVec operator-(CPVec b)const{return {x-b.x,y-b.y,z-b.z};}CPVec operator*(double s)const{return {x*s,y*s,z*s};}};
double cp_dot(CPVec a,CPVec b){return a.x*b.x+a.y*b.y+a.z*b.z;}
CPVec cp_cross(CPVec a,CPVec b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
CPVec cp_vec(float4 a){return {a.x,a.y,a.z};}
CPVec cp_rotate(float4 q,CPVec v,bool inverse=false){double length=std::sqrt(double(q.x)*q.x+double(q.y)*q.y+double(q.z)*q.z+double(q.w)*q.w);CPVec u=cp_vec(q)*(1/length);if(inverse)u=u*(-1);double w=q.w/length;return v+cp_cross(u,v)*(2*w)+cp_cross(u,cp_cross(u,v))*2;}
struct CPFrozen {uint4 info;float4 bc;CPVec local_normal,local_correction;double plane;float rest;};
std::vector<CPFrozen> cp_frozen;
std::vector<uint4> cp_seed_info,cp_seed_tets;std::vector<float4> cp_seed_bc,cp_seed_point,cp_seed_normal,cp_seed_position;
float4 cp_seed_body[2]={};bool cp_seed_loaded=false;
bool cp_canonical(float4& v){double s=double(v.x)+v.y+v.z+v.w;float* w=&v.x;int largest=0;for(int k=0;k<4;++k){if(!std::isfinite(w[k])||w[k]<0)return false;if(w[k]>w[largest])largest=k;}if(!(s>0)||std::fabs(s-1)>1e-4)return false;for(int k=0;k<4;++k)w[k]=float(double(w[k])/s);double other=0;for(int k=0;k<4;++k)if(k!=largest)other+=w[k];w[largest]=float(1-other);return w[largest]>0;}
CPVec cp_point(uint4 info,float4 weights,const std::vector<uint4>& ti,const std::vector<float4>& p){uint4 ids=ti[info.z&0xfffff];double sum=double(weights.x)+weights.y+weights.z+weights.w;CPVec point{};for(int k=0;k<4;++k)point=point+cp_vec(p[(&ids.x)[k]])*(double((&weights.x)[k])/sum);return point;}
bool cp_read(void* dst,CUdeviceptr src,size_t n){return !n || cuMemcpyDtoH(dst,src,n)==CUDA_SUCCESS;}
bool cp_write(CUdeviceptr dst,const void* src,size_t n){return !n || cuMemcpyHtoD(dst,src,n)==CUDA_SUCCESS;}
bool cp_prepare(void** params){
 if(!cp_policy)return true;
 std::lock_guard<std::mutex> g(cp_lock);
 if(!params || cp_frame<0 || cuCtxSynchronize()!=CUDA_SUCCESS)return false;
 CUdeviceptr a[13]={};for(int i=0;i<13;++i)if(i!=10 && i!=12)a[i]=*static_cast<CUdeviceptr*>(params[i]);
 unsigned c=0;if(!cp_read(&c,a[5],4) || c>16384 || !*static_cast<unsigned char*>(params[12]))return false;
 std::vector<uint4> info(c);std::vector<float4> bc(c),point(c),normal(c);
 if(!cp_read(info.data(),a[4],16*c)||!cp_read(bc.data(),a[3],16*c)||!cp_read(point.data(),a[1],16*c)||!cp_read(normal.data(),a[2],16*c))return false;
 std::vector<float4> p(nodes);std::vector<uint4> ti;float4 frame[2]={};unsigned sid=0;CUdeviceptr body=0;
 if(c){
  if((info[0].y&1)||info[0].x==~0u)return false;
  unsigned softid=info[0].z>>20;body=a[0]+432*size_t(softid);unsigned char meta[432];CUdeviceptr ids=0,frames=0;
  if(!cp_read(meta,body,432)||field<unsigned>(meta,372)!=nodes||field<CUdeviceptr>(meta,96)!=positions)return false;
  unsigned nt=field<unsigned>(meta,376);if(!nt||nt>1000000)return false;ti.resize(nt);
  if(!cp_read(ti.data(),field<CUdeviceptr>(meta,80),nt*16)||!cp_read(p.data(),positions,nodes*16)||
     !cp_read(&ids,a[7]+272,8)||!cp_read(&sid,ids+info[0].x*4,4)||sid>=64||!cp_read(&frames,a[8]+168,8)||!cp_read(frame,frames+32*sid,32))return false;
  for(unsigned i=0;i<c;++i)if(info[i].x!=info[0].x || info[i].y!=info[0].y || (info[i].z>>20)!=softid || (info[i].z&0xfffff)>=nt)return false;
 }
 long offset=std::ftell(cp_blob);
 auto put=[&](const void* v,size_t n){return !n||std::fwrite(v,1,n,cp_blob)==n;};
 if(!put(info.data(),c*16)||!put(bc.data(),c*16)||!put(point.data(),c*16)||!put(normal.data(),c*16)||!put(p.data(),nodes*16)||!put(ti.data(),ti.size()*16)||!put(frame,32))return false;
 bool unified=cp_policy==2||cp_policy==4,frozen=cp_policy>=3&&cp_frame>=cp_freeze_frame,seeded=cp_seed_loaded&&cp_frame==cp_freeze_frame;
 double maxshift=0,maxsum=0,seed_difference=0;
 if(seeded){
  if(!c||c!=cp_seed_info.size()||ti.size()!=cp_seed_tets.size()||std::memcmp(ti.data(),cp_seed_tets.data(),ti.size()*16))return false;
  for(size_t k=0;k<nodes;++k){CPVec change=cp_vec(p[k])-cp_vec(cp_seed_position[k]);seed_difference=std::fmax(seed_difference,std::sqrt(cp_dot(change,change)));if(p[k].w!=cp_seed_position[k].w)return false;}
  if(seed_difference>1e-7)return false;
  for(unsigned i=0;i<c;++i){
   uint4 ci=cp_seed_info[i];ci.w&=0x7fffffffu;
   if(info[i].x!=ci.x||info[i].y!=ci.y||(info[i].z>>20)!=(ci.z>>20))return false;
   info[i]=ci;bc[i]=cp_seed_bc[i];
   CPVec seed_p=cp_vec(cp_seed_point[i]),seed_n=cp_vec(cp_seed_normal[i]);
   CPVec local_correction=cp_rotate(cp_seed_body[0],seed_p-cp_point(ci,bc[i],ti,cp_seed_position),true);
   CPVec fresh=cp_point(ci,bc[i],ti,p)+cp_rotate(frame[0],local_correction);
   CPVec nw=cp_rotate(frame[0],cp_rotate(cp_seed_body[0],seed_n,true));
   double plane=cp_dot(seed_n,seed_p-cp_vec(cp_seed_body[1]))-cp_seed_normal[i].w;
   point[i]={float(fresh.x),float(fresh.y),float(fresh.z),cp_seed_point[i].w};
   normal[i]={float(nw.x),float(nw.y),float(nw.z),float(cp_dot(nw,fresh-cp_vec(frame[1]))-plane)};
  }
 }
 if(unified)for(unsigned i=0;i<c;++i){
  if(!cp_canonical(bc[i]))return false;
  CPVec fresh=cp_point(info[i],bc[i],ti,p),change=fresh-cp_vec(point[i]);
  maxshift=std::fmax(maxshift,std::sqrt(cp_dot(change,change)));
  normal[i].w=float(double(normal[i].w)+cp_dot(cp_vec(normal[i]),change));
  point[i].x=float(fresh.x);point[i].y=float(fresh.y);point[i].z=float(fresh.z);
 }
 if(frozen){
  if(cp_frame==cp_freeze_frame){
   if(!c || !cp_frozen.empty())return false;
   for(unsigned i=0;i<c;++i){
    CPVec nl=cp_rotate(frame[0],cp_vec(normal[i]),true);
    double plane=cp_dot(cp_vec(normal[i]),cp_vec(point[i])-cp_vec(frame[1]))-normal[i].w;
    // This experiment freezes the top face of the fixture's 0.16x0.16x0.02 box.
    // Reject a different face/shape instead of extrapolating arbitrary contacts.
    if(std::fabs(nl.x)>1e-5||std::fabs(nl.y)>1e-5||std::fabs(nl.z-1)>1e-5||std::fabs(plane-.01)>2e-6)return false;
    auto ci=info[i];ci.w&=0x7fffffffu;
    cp_frozen.push_back({ci,bc[i],nl,cp_rotate(frame[0],cp_vec(point[i])-cp_point(ci,bc[i],ti,p),true),plane,point[i].w});
   }
  }else{
   if(c!=cp_frozen.size()||!c)return false; // Never enlarge native allocations.
   for(unsigned i=0;i<c;++i){auto f=cp_frozen[i];
    if(info[i].x!=f.info.x||info[i].y!=f.info.y||(info[i].z>>20)!=(f.info.z>>20))return false;
    info[i]=f.info;bc[i]=f.bc;
    CPVec fresh=cp_point(info[i],bc[i],ti,p)+cp_rotate(frame[0],f.local_correction),nw=cp_rotate(frame[0],f.local_normal);
    CPVec local=cp_rotate(frame[0],fresh-cp_vec(frame[1]),true);
    if(std::fabs(local.x)>.079||std::fabs(local.y)>.079)return false;
    double pen=cp_dot(nw,fresh-cp_vec(frame[1]))-f.plane;
    point[i]={float(fresh.x),float(fresh.y),float(fresh.z),f.rest};normal[i]={float(nw.x),float(nw.y),float(nw.z),float(pen)};
   }
  }
 }
 // The seeded factorial experiment uses exactly the same analytic box-face
 // geometry evaluation in dynamic and frozen arms. Otherwise manifold choice
 // would be confounded with native-vs-custom gap/normal evaluation.
 bool common_plane=cp_seed_loaded&&cp_frame>=cp_freeze_frame;
 if(common_plane)for(unsigned i=0;i<c;++i){
  CPVec fresh=cp_point(info[i],bc[i],ti,p),nw=cp_rotate(frame[0],{0,0,1});
  CPVec local=cp_rotate(frame[0],fresh-cp_vec(frame[1]),true);
  if(point[i].w!=0.f||std::fabs(local.x)>.079||std::fabs(local.y)>.079)return false;
  point[i]={float(fresh.x),float(fresh.y),float(fresh.z),0.f};
  normal[i]={float(nw.x),float(nw.y),float(nw.z),float(cp_dot(nw,fresh-cp_vec(frame[1]))-.01)};
 }
 for(unsigned i=0;i<c;++i)maxsum=std::fmax(maxsum,std::fabs(double(bc[i].x)+bc[i].y+bc[i].z+bc[i].w-1));
 if((unified||frozen||seeded||common_plane)&&(!cp_write(a[4],info.data(),c*16)||!cp_write(a[3],bc.data(),c*16)||!cp_write(a[1],point.data(),c*16)||!cp_write(a[2],normal.data(),c*16)))return false;
 long applied=std::ftell(cp_blob);
 if(!put(info.data(),c*16)||!put(bc.data(),c*16)||!put(point.data(),c*16)||!put(normal.data(),c*16))return false;
 if(std::fprintf(cp_log,"{\"frame\":%d,\"contacts\":%u,\"nodes\":%zu,\"tets\":%zu,\"sid\":%u,\"offset\":%ld,\"bytes\":%zu,\"applied_offset\":%ld,\"policy\":%d,\"frozen\":%s,\"seed_applied\":%s,\"seed_position_max_difference_m\":%.17g,\"max_normalize_point_shift_m\":%.17g,\"max_weight_sum_defect\":%.17g}\n",cp_frame,c,nodes,ti.size(),sid,offset,c*64+nodes*16+ti.size()*16+32,applied,cp_policy,frozen?"true":"false",seeded?"true":"false",seed_difference,maxshift,maxsum)<0)return false;
 ++cp_prepares;return true;
}
}
extern "C" int observer_contact_policy_start(const char* path,int policy,int freeze_frame){
 if(!enabled || mode!=9 || cp_log || policy<1 || policy>4 || freeze_frame<0)return -1;
 cp_log=std::fopen((std::string(path)+".jsonl").c_str(),"wx");cp_blob=std::fopen((std::string(path)+".bin").c_str(),"wx");
 if(!cp_log || !cp_blob)return -2;
 if(policy==2||policy==4){CUresult c=cuCtxPushCurrent(context);if(c!=CUDA_SUCCESS)return int(c);c=cuModuleGetFunction(&cp_countfn,module,"complete_contact_support");CUcontext previous;cuCtxPopCurrent(&previous);if(c!=CUDA_SUCCESS)return int(c);}
 cp_policy=policy;cp_freeze_frame=freeze_frame;return 0;
}
extern "C" void observer_frame(int frame){cp_frame=frame;}
extern "C" int observer_contact_seed_load(const char* path){
 if(!cp_policy||cp_seed_loaded||cp_prepares)return -1;
 FILE* f=std::fopen(path,"rb");if(!f)return -2;unsigned char header[64];
 auto get=[&](void* dst,size_t n){return !n||std::fread(dst,1,n,f)==n;};
 if(!get(header,64)||std::memcmp(header,"CPCSEED1",8)){std::fclose(f);return -3;}
 unsigned c=field<unsigned>(header,8),n=field<unsigned>(header,12),nt=field<unsigned>(header,16),frame=field<unsigned>(header,20);
 if(!c||c>16384||n!=nodes||!nt||nt>1000000||frame!=unsigned(cp_freeze_frame)){std::fclose(f);return -4;}
 std::memcpy(cp_seed_body,header+24,32);cp_seed_info.resize(c);cp_seed_bc.resize(c);cp_seed_point.resize(c);cp_seed_normal.resize(c);cp_seed_position.resize(n);cp_seed_tets.resize(nt);
 bool good=get(cp_seed_info.data(),c*16)&&get(cp_seed_bc.data(),c*16)&&get(cp_seed_point.data(),c*16)&&get(cp_seed_normal.data(),c*16)&&get(cp_seed_position.data(),n*16)&&get(cp_seed_tets.data(),nt*16)&&std::fgetc(f)==EOF;
 std::fclose(f);if(!good)return -5;
 for(auto t:cp_seed_tets)if(t.x>=nodes||t.y>=nodes||t.z>=nodes||t.w>=nodes)return -6;
 for(unsigned i=0;i<c;++i){float4 b=cp_seed_bc[i];if((cp_seed_info[i].z&0xfffff)>=nt||!cp_canonical(b))return -7;}
 for(auto x:cp_seed_position)if(!std::isfinite(x.x)||!std::isfinite(x.y)||!std::isfinite(x.z)||!(x.w>0))return -8;
 cp_seed_loaded=true;return 0;
}
extern "C" unsigned long long observer_contact_policy_queries(){return cp_queries;}
extern "C" unsigned long long observer_contact_policy_stop(){cp_policy=0;if(cp_log){if(std::fclose(cp_log))++errors;cp_log=nullptr;}if(cp_blob){if(std::fclose(cp_blob))++errors;cp_blob=nullptr;}return cp_prepares;}
