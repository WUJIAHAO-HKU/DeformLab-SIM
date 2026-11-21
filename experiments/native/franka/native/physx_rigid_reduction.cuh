#include "physx_rigid_reduction_state.h"
// Read each native per-contact delta BEFORE the prefix-sum kernel overwrites it.
// Only linear velocity is changed. Angular and all position updates stay native.
extern "C" __global__ void capture_rigid_reduction(const uint4* info,const unsigned* count,
 char* prep,char* core,char* shared,const float4* raw,const unsigned* references,
 RigidReductionState* state,unsigned* rejected,int policy){
 unsigned sid=threadIdx.x,c=*count,nb=field<unsigned>(core,268);
 if(nb>64||c>16384){if(!sid)atomicAdd(rejected,1u);return;}
 if(sid>=nb||!c)return;
 unsigned* ids=field<unsigned*>(prep,272);unsigned offset=field<unsigned>(core,276);
 float4* pool=field<float4*>(shared,160);auto& s=state[sid];
 double3 sum={};unsigned active=0,node=~0u;
 for(unsigned i=0;i<c;++i){auto ci=info[i];
  if(ci.x==~0u||(ci.y&1u)){if(!sid)atomicAdd(rejected,1u);return;}
  if(ids[ci.x]!=sid)continue;
  if(node!=~0u&&node!=ci.x){atomicAdd(rejected,1u);return;}node=ci.x;
  bool collision=(ci.w&0x80000000u)!=0;float4 v=raw[i];
  if(v.w!=(collision?1.f:0.f)){atomicAdd(rejected,1u);return;}
  if(collision){sum.x+=double(v.x);sum.y+=double(v.y);sum.z+=double(v.z);++active;}
 }
 if(active!=references[sid]){atomicAdd(rejected,1u);return;}
 if(!active)return;
 float4 before=pool[offset+sid];
 if(s.pending || (s.initialized&&s.rigid_node!=node)){atomicAdd(rejected,1u);return;}
 if(s.initialized){
  double3 change={double(before.x)-s.published.x,double(before.y)-s.published.y,double(before.z)-s.published.z};
  s.continuity_error.x=fmax(s.continuity_error.x,fabs(change.x));s.continuity_error.y=fmax(s.continuity_error.y,fabs(change.y));s.continuity_error.z=fmax(s.continuity_error.z,fabs(change.z));
  // Compensated carry is allowed only if native intervening stages preserved
  // this free body's published physical linear velocity exactly.
  if(policy==2&&(change.x!=0||change.y!=0||change.z!=0)){++s.faults;atomicAdd(rejected,1u);return;}
 }
 if(policy==1)s.residual={0,0,0};
 s.before=before;s.increment={sum.x/active,sum.y/active,sum.z/active};
 s.rigid_node=node;s.pending=1;s.initialized=1;++s.samples;
}
extern "C" __global__ void publish_rigid_reduction(char* core,char* shared,RigidReductionState* state,unsigned* rejected,int policy){
 unsigned sid=threadIdx.x,nb=field<unsigned>(core,268);if(nb>64){if(!sid)atomicAdd(rejected,1u);return;}if(sid>=nb)return;
 auto& s=state[sid];if(!s.pending)return;
 float4* pool=field<float4*>(shared,160);unsigned offset=field<unsigned>(core,276);float4 out=pool[offset+sid];
 s.native_roundoff_sum.x+=(double(out.x)-s.before.x)-s.increment.x;
 s.native_roundoff_sum.y+=(double(out.y)-s.before.y)-s.increment.y;
 s.native_roundoff_sum.z+=(double(out.z)-s.before.z)-s.increment.z;
 double3 exact={double(s.before.x)+s.residual.x+s.increment.x,double(s.before.y)+s.residual.y+s.increment.y,double(s.before.z)+s.residual.z+s.increment.z};
 out.x=float(exact.x);out.y=float(exact.y);out.z=float(exact.z);
 s.residual=policy==2?make_double3(exact.x-out.x,exact.y-out.y,exact.z-out.z):make_double3(0,0,0);
 // Preserve the angular X value in W; no torque or momentum projection.
 pool[offset+sid]=out;s.published=out;s.pending=0;
}
