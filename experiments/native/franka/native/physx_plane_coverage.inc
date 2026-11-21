// Pinned, opt-in Cartesian coupon experiment. Reuses native contact capacity;
// does not move vertices, change masses/material/drive, or allocate contacts.
// Unsupported shapes, frames or insufficient capacity fail closed.
#include <algorithm>
#include <map>
#include "contact_slot_budget.h"
#include "contact_surface_selection.h"
namespace {
int pc_policy=0;FILE* pc_log=nullptr;FILE* pc_blob=nullptr;
unsigned long long pc_prepares=0,pc_rejections=0;std::vector<unsigned> pc_faces[2];
double pc_planes[2]={NAN,NAN};
double pc_band=NAN,pc_qstart=NAN;CUdeviceptr pc_dofs=0;
std::vector<unsigned> pc_omitted[2];FILE* pc_guard_log=nullptr;FILE* pc_guard_blob=nullptr;
bool pc_guard_armed=false;unsigned long long pc_guard_records=0;
bool pc_inputs_usable=true;
bool pc_fail(const char* reason){++pc_rejections;std::fprintf(stderr,"Plane coverage rejected: %s frame=%d\n",reason,cp_frame);if(pc_log){std::fprintf(pc_log,"{\"frame\":%d,\"phase\":\"rejected\",\"reason\":\"%s\",\"accepted\":false}\n",cp_frame,reason);std::fflush(pc_log);}if(pc_blob)std::fflush(pc_blob);return false;}
bool pc_prepare(void** params){
 if(!pc_policy)return true;
 pc_inputs_usable=true;
 if(!params||cp_frame<0||!std::isfinite(pc_planes[0])||!std::isfinite(pc_planes[1])||cuCtxSynchronize()!=CUDA_SUCCESS)return pc_fail("launch state");
 CUdeviceptr a[6];for(int k=0;k<6;++k)a[k]=*static_cast<CUdeviceptr*>(params[k]);
 unsigned c=0;if(!cp_read(&c,a[5],4)||c>16384||!*static_cast<unsigned char*>(params[12]))return pc_fail("count/TGS");
 std::vector<uint4> info(c);std::vector<float4> bc(c),point(c),normal(c),p(nodes);
 if(!cp_read(info.data(),a[4],16*c)||!cp_read(bc.data(),a[3],16*c)||!cp_read(point.data(),a[1],16*c)||!cp_read(normal.data(),a[2],16*c)||!cp_read(p.data(),positions,16*nodes))return pc_fail("readback");
 long raw_offset=std::ftell(pc_blob);
 auto put=[&](const void* v,size_t n){return !n||std::fwrite(v,1,n,pc_blob)==n;};
 if(!put(info.data(),c*16)||!put(bc.data(),c*16)||!put(point.data(),c*16)||!put(normal.data(),c*16)||!put(p.data(),nodes*16))return pc_fail("capture");
 std::fprintf(pc_log,"{\"frame\":%d,\"phase\":\"input\",\"contacts\":%u,\"nodes\":%zu,\"raw_offset\":%ld,\"planes_m\":[%.17g,%.17g]}\n",cp_frame,c,nodes,raw_offset,pc_planes[0],pc_planes[1]);
 std::vector<uint4> ti;unsigned softid=0,nt=0;
 if(c){
  softid=info[0].z>>20;unsigned char meta[432];
  if(!cp_read(meta,a[0]+size_t(softid)*432,432)||field<unsigned>(meta,372)!=nodes||field<CUdeviceptr>(meta,96)!=positions)return pc_fail("soft ABI");
  nt=field<unsigned>(meta,376);if(!nt||nt>1000000)return pc_fail("tet count");ti.resize(nt);
  if(!cp_read(ti.data(),field<CUdeviceptr>(meta,80),16*nt))return pc_fail("tet read");
 }
 struct Face {std::vector<unsigned> slots;std::vector<double> planes;uint4 actor{};};Face faces[2];
 for(unsigned i=0;i<c;++i){
  auto ni=normal[i];auto ci=info[i];
  if((ci.z>>20)!=softid||(ci.z&0xfffff)>=nt||point[i].w!=0.f||!std::isfinite(ni.w))return pc_fail("invalid contact metadata");
  int side=ci.x==~0u&&ci.y==0?0:1;double sign=side==0?1.:-1.;auto& face=faces[side];
  // Geometry comes from the modelled box pose, not noisy GJK normal components.
  // The independent model/contact gap agreement retains the 2 micrometre gate.
  // At nearly zero separation GJK's direction can be ill-conditioned. The
  // fixture actor, validated model pose, face footprint and model gap identify
  // the face; its exact model normal is used in both experimental arms.
  if(!std::isfinite(ni.x)||!std::isfinite(ni.y)||!std::isfinite(ni.z))return pc_fail("nonfinite contact normal");
  // Raw GJK points can be rigid-box vertices outside the soft-body footprint.
  // Validate footprint on the actual barycentric/node point below instead.
  if(std::fabs(sign*(double(point[i].z)-pc_planes[side])-ni.w)>2e-6)return pc_fail("model/contact gap mismatch");
  if(face.slots.empty())face.actor=ci;
  if(face.actor.x!=ci.x||face.actor.y!=ci.y)return pc_fail("multiple actors per face");
  if((side==0&&(ci.x!=~0u||ci.y!=0))||(side==1&&!(ci.y&1)))return pc_fail("expected static bench/articulated plate");
  face.slots.push_back(i);face.planes.push_back(double(point[i].z)-double(ni.w)/sign);
 }
 unsigned shortfall=0;double plane[2]={NAN,NAN},spread[2]={0,0};
 std::size_t original_counts[2]={faces[0].slots.size(),faces[1].slots.size()};
 for(int side=0;side<2;++side){auto& face=faces[side];if(face.slots.empty())continue;
  std::sort(face.planes.begin(),face.planes.end());plane[side]=pc_planes[side];spread[side]=face.planes.back()-face.planes.front();
  if(spread[side]>2e-6||(side==0&&std::fabs(plane[side]-.12)>2e-6)||(side==1&&(plane[side]<.12||plane[side]>.3)))return pc_fail("plane mismatch");
 }
 std::array<std::size_t,2> borrowed={0,0};
 std::vector<unsigned> selected[2]={pc_faces[0],pc_faces[1]};
 std::vector<double> gaps[2];
 if(pc_policy==4){
  if(!std::isfinite(pc_band)||!std::isfinite(pc_qstart))return pc_fail("guard frame configuration");
  for(int side=0;side<2;++side)for(unsigned id:pc_faces[side]){
   double gap=(side==0?1.:-1.)*(double(p[id].z)-pc_planes[side]);
   if(!std::isfinite(gap))return pc_fail("nonfinite surface gap");gaps[side].push_back(gap);
  }
 }
 if(pc_policy==3||pc_policy==4){
  std::array<std::vector<unsigned>,2> slots={faces[0].slots,faces[1].slots};
  std::array<std::size_t,2> required={slots[0].empty()?0:pc_faces[0].size(),slots[1].empty()?0:pc_faces[1].size()};
  if(pc_policy==4&&required[0]+required[1]>c){
   for(int side=0;side<2;++side)if(!slots[side].empty())required[side]=std::max<std::size_t>(1,std::count_if(gaps[side].begin(),gaps[side].end(),[](double g){return g<=pc_band;}));
  }
  if(required[0]+required[1]>c)shortfall=required[0]+required[1]-c;
  if(!rebalance_contact_slots(slots,required,borrowed)&&!shortfall)return pc_fail("contact slot ownership");
  if(!shortfall){faces[0].slots=std::move(slots[0]);faces[1].slots=std::move(slots[1]);}
 }
 // Record a rejected capacity observation before returning, never overwrite
 // beyond native storage or silently run an incomplete candidate.
 if((pc_policy==3||pc_policy==4)&&shortfall){
  std::fprintf(pc_log,"{\"frame\":%d,\"policy\":%d,\"contacts\":%u,\"capacity_shortfall\":%u,\"raw_offset\":%ld,\"accepted\":false}\n",cp_frame,pc_policy,c,shortfall,raw_offset);std::fflush(pc_log);
  return pc_fail("insufficient existing contact slots");
 }
 if(pc_policy==4)for(int side=0;side<2;++side){
  if(faces[side].slots.empty()){selected[side].clear();continue;}
  if(!select_contact_surface_nodes(pc_faces[side],gaps[side],faces[side].slots.size(),pc_band,selected[side]))return pc_fail("surface band capacity");
 }
 std::vector<unsigned> node_tet(nodes,~0u),node_corner(nodes,~0u);
 if(pc_policy==3||pc_policy==4)for(unsigned t=0;t<nt;++t)for(unsigned k=0;k<4;++k){unsigned id=(&ti[t].x)[k];if(id>=nodes)return pc_fail("tet node");if(node_tet[id]==~0u){node_tet[id]=t;node_corner[id]=k;}}
 double max_gap[2]={0,0};
 if(pc_policy>=2)for(int side=0;side<2;++side){auto& face=faces[side];double sign=side==0?1.:-1.;
  for(unsigned j=0;j<face.slots.size();++j){unsigned i=face.slots[j];CPVec x{};
   if(pc_policy==3||pc_policy==4){unsigned id=selected[side][j%selected[side].size()];if(node_tet[id]==~0u)return pc_fail("unmapped surface node");
    info[i]=face.actor;info[i].z=(softid<<20)|node_tet[id];info[i].w&=0x7fffffffu;
    bc[i]={0,0,0,0};(&bc[i].x)[node_corner[id]]=1.f;x=cp_vec(p[id]);
   }else{if(!cp_canonical(bc[i]))return pc_fail("invalid weights");x=cp_point(info[i],bc[i],ti,p);}
   if(std::fabs(x.x)>.075||std::fabs(x.y)>.075)return pc_fail("point outside coupon face");
   double gap=sign*(x.z-plane[side]);max_gap[side]=std::fmax(max_gap[side],std::fabs(gap));
   point[i]={float(x.x),float(x.y),float(x.z),0.f};normal[i]={0,0,float(sign),float(gap)};
  }
 }
 if(pc_policy>=2){
  pc_inputs_usable=false;
  if(!cp_write(a[4],info.data(),c*16)||!cp_write(a[3],bc.data(),c*16)||!cp_write(a[1],point.data(),c*16)||!cp_write(a[2],normal.data(),c*16))return pc_fail("publication");
  pc_inputs_usable=true;
 }
 if(pc_policy==4){
  for(int side=0;side<2;++side){pc_omitted[side].clear();
   // Monitor only nodes omitted by this policy from an existing contact pair.
   // No-pair broadphase behavior remains a separate coverage check.
   if(faces[side].slots.empty())continue;
   for(unsigned id:pc_faces[side])if(std::find(selected[side].begin(),selected[side].end(),id)==selected[side].end())pc_omitted[side].push_back(id);
  }
  pc_guard_armed=!pc_omitted[0].empty()||!pc_omitted[1].empty();
 }
 long applied_offset=std::ftell(pc_blob);
 if(!put(info.data(),c*16)||!put(bc.data(),c*16)||!put(point.data(),c*16)||!put(normal.data(),c*16))return pc_fail("applied capture");
 if(std::fprintf(pc_log,"{\"frame\":%d,\"policy\":%d,\"contacts\":%u,\"nodes\":%zu,\"raw_offset\":%ld,\"applied_offset\":%ld,\"side_counts\":[%zu,%zu],\"original_side_counts\":[%zu,%zu],\"borrowed_slots\":[%zu,%zu],\"capacity_shortfall\":%u,\"plane_spread_m\":[%.17g,%.17g],\"max_gap_m\":[%.17g,%.17g],\"accepted\":true}\n",cp_frame,pc_policy,c,nodes,raw_offset,applied_offset,faces[0].slots.size(),faces[1].slots.size(),original_counts[0],original_counts[1],borrowed[0],borrowed[1],shortfall,spread[0],spread[1],max_gap[0],max_gap[1])<0)return pc_fail("log");
 ++pc_prepares;return true;
}
void pc_stop(){pc_policy=0;pc_planes[0]=pc_planes[1]=NAN;pc_band=pc_qstart=NAN;pc_dofs=0;pc_guard_armed=false;
 for(FILE** f:{&pc_log,&pc_blob,&pc_guard_log,&pc_guard_blob})if(*f){if(std::fclose(*f))++errors;*f=nullptr;}
 for(auto& f:pc_faces)f.clear();for(auto& f:pc_omitted)f.clear();}
}
extern "C" int observer_plane_coverage_start(const char* path,int policy,const unsigned* bottom,unsigned nb,const unsigned* top,unsigned nt){
 if(!enabled||mode!=6||pc_policy||pc_log||cp_policy||policy<1||policy>4||!bottom||!top||!nb||!nt)return -1;
 for(unsigned i=0;i<nb;++i)if(bottom[i]>=nodes)return -2;
 for(unsigned i=0;i<nt;++i)if(top[i]>=nodes)return -2;
 pc_faces[0].assign(bottom,bottom+nb);pc_faces[1].assign(top,top+nt);pc_prepares=0;pc_rejections=0;
 pc_log=std::fopen((std::string(path)+".jsonl").c_str(),"wx");pc_blob=std::fopen((std::string(path)+".bin").c_str(),"wx");
 if(policy==4){pc_guard_log=std::fopen((std::string(path)+"-guard.jsonl").c_str(),"wx");pc_guard_blob=std::fopen((std::string(path)+"-guard.bin").c_str(),"wx");pc_guard_records=0;
  if(!pc_guard_log||!pc_guard_blob){pc_stop();return -3;}}
 if(!pc_log||!pc_blob){pc_stop();return -3;}pc_policy=policy;return 0;
}
extern "C" unsigned long long observer_plane_coverage_stop(){pc_stop();return pc_prepares;}
extern "C" unsigned long long observer_plane_coverage_errors(){return pc_rejections;}
extern "C" int observer_plane_coverage_frame(int frame,double lower,double upper){
 if(!pc_policy||frame<1||!std::isfinite(lower)||!std::isfinite(upper)||!(upper>lower))return -1;
 cp_frame=frame;pc_planes[0]=lower;pc_planes[1]=upper;pc_guard_armed=false;return 0;
}
#include "physx_plane_guard.inc"
