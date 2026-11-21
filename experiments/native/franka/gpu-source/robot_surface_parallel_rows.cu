#include <cstdio>
// Device-only fusion of the verified row arithmetic; generated helpers retain
// the original arithmetic and add only a chunk index for append/restore loops.
#include "robot_surface_stream_state.h"
#include "robot_surface_device_rows.inc"
template<class T> static __device__ T* ptr(uint64_t value){return reinterpret_cast<T*>(value);}
static __device__ void reject(SurfaceStreamState* s,unsigned code){
 // Quarantine immediately: subsequent native kernels must read zero rows.
 atomicExch(ptr<unsigned>(s->count),0u);
 atomicCAS(ptr<unsigned long long>(s->telemetry)+4,0ull,(unsigned long long)code);
 atomicCAS(ptr<unsigned>(s->header)+6,0u,code);
}
static __device__ bool failed(SurfaceStreamState* s){
 return ptr<unsigned>(s->header)[6]!=0||ptr<unsigned long long>(s->telemetry)[4]!=0;
}
static __device__ void guards(SurfaceStreamState* s){
 const uint64_t* tails=ptr<uint64_t>(s->guards);
 for(unsigned i=threadIdx.x;i<s->guard_count*16;i+=blockDim.x){
  if(ptr<unsigned>(tails[i/16])[i%16]!=0xa5a5a5a5u)reject(s,804);
 }
}
template<class T> static __device__ T field_at(uint64_t address,unsigned offset){return *ptr<T>(address+offset);}
static __device__ void native_bindings(SurfaceStreamState* s,unsigned round){
 if(!s->native_validation)return;
 uint64_t soft=s->native_binding[0],prep=s->native_binding[1],core=s->native_binding[2],art=s->native_binding[3];
 if(threadIdx.x==0){
  if(field_at<unsigned>(soft,372)!=s->nodes||field_at<unsigned>(soft,376)!=s->tets||field_at<uint64_t>(soft,80)!=s->input[5]||field_at<uint64_t>(soft,96)!=s->input[6]||field_at<uint64_t>(soft,128)!=s->input[7]||field_at<uint64_t>(soft,88)!=s->native_binding[4])reject(s,405);
  if(field_at<uint64_t>(prep,272)!=s->native_binding[5])reject(s,409);
 }
 unsigned nb=field_at<unsigned>(core,268),art_count=field_at<unsigned>(art,44),links=field_at<unsigned>(art,248);
 if(!nb||nb>64||art_count>64||links>128){if(threadIdx.x==0)reject(s,408);return;}
 for(unsigned i=threadIdx.x;i<s->native_count;i+=blockDim.x){
  uint4 info=ptr<uint4>(s->input[0])[i];unsigned id=~0u;
  if(info.x!=~0u){
   if(info.x>=s->map_count){reject(s,409);continue;}
   unsigned mapped=ptr<unsigned>(s->native_binding[5])[info.x];
   if(info.y&1u){if(mapped>=art_count||(info.y>>1)>=links){reject(s,410);continue;}id=nb+mapped*links+(info.y>>1);}
   else {if(mapped>=nb){reject(s,411);continue;}id=mapped;}
  }
  if(round==0)ptr<unsigned>(s->input[1])[i]=id;else if(ptr<unsigned>(s->input[1])[i]!=id)reject(s,412);
 }
}
#include "robot_surface_parallel_plan.inc"
#include "component_dense_plan.inc"
#ifndef COMPONENT_DENSE_POLICY
#error Explicit component policy required
#endif
static_assert(COMPONENT_DENSE_POLICY==0 || COMPONENT_DENSE_POLICY==1,"policy");
#ifndef COMPONENT_ROW_DIAGNOSTICS
#error Explicit diagnostic policy required
#endif
static_assert(COMPONENT_ROW_DIAGNOSTICS==0 || COMPONENT_ROW_DIAGNOSTICS==1,"diagnostics");
extern "C" { __device__ unsigned long long component_trace[256*24]; }
static __device__ void component_record(SurfaceStreamState* s,unsigned stage){
 if(threadIdx.x)return;
 unsigned* h=ptr<unsigned>(s->header);unsigned round=h[2];if(round>=256){reject(s,842);return;}
#if COMPONENT_ROW_DIAGNOSTICS
 unsigned* p=ptr<unsigned>(s->plan);unsigned n=p[1],nc=p[2];auto t=component_trace+24*round;
 if(stage==0){t[0]=s->frame;t[1]=round;t[2]=COMPONENT_DENSE_POLICY;t[3]++;t[4]=n;t[5]=nc-n;t[23]=s->capacity;}
 if(stage==1)t[6]++;if(stage==2)t[7]++;
 double sum=0;unsigned positive=0,active=0,posgapactive=0;float mingap=INFINITY,maxgap=-INFINITY;unsigned posgap=0,neggap=0;unsigned long long identity=0;
 for(unsigned i=n;i<nc;i++){
  float l=ptr<float>(s->lambda)[i];sum+=double(l);positive+=l>0;
  unsigned a=(ptr<uint4>(s->info)[i].w>>31);active+=a;
  float gap=ptr<float>(s->constraints)[(i/32)*800+384+4*(i%32)+3];
  if(gap<mingap)mingap=gap;if(gap>maxgap)maxgap=gap;posgap+=gap>0;neggap+=gap<0;posgapactive+=(gap>0&&a);
  identity^=(static_cast<unsigned long long>(ptr<uint4>(s->selected)[i-n].x)+1)*0x9e3779b97f4a7c15ull;
  if(l<0||!isfinite(l))t[22]++;
 }
 if(stage==0){t[8]=positive;t[10]=active;t[13]=identity;t[14]=__double_as_longlong(sum);t[16]=__float_as_uint(maxgap);t[17]=__float_as_uint(mingap);t[18]=posgap;t[19]=neggap;}
 if(stage==1){t[11]=active;t[20]=posgapactive;t[21]=posgap-posgapactive;}
 if(stage==2){t[9]=positive;t[12]=active;t[15]=__double_as_longlong(sum);}
#endif
}

extern "C" __global__ void surface_stream_reset(SurfaceStreamState* s,unsigned frame){
#if COMPONENT_ROW_DIAGNOSTICS
 for(unsigned i=threadIdx.x;i<256*24;i+=blockDim.x)component_trace[i]=0;
#endif
 __syncthreads();
 unsigned* h=ptr<unsigned>(s->header);
 device_robot_surface_history_reset(h,ptr<float>(s->cache),ptr<unsigned>(s->seen),ptr<unsigned>(s->active),s->nodes,s->boundary_count,frame);
 __syncthreads();
 if(threadIdx.x==0){
  if(h[6])reject(s,20000+h[6]);
  else ptr<unsigned long long>(s->telemetry)[3]++;
 }
}
extern "C" __global__ void surface_stream_prepare(SurfaceStreamState* s,unsigned round){
 unsigned* h=ptr<unsigned>(s->header);unsigned* plan=ptr<unsigned>(s->plan);
 if(threadIdx.x==0){
  *ptr<unsigned>(s->count)=0;
  unsigned observed=*ptr<unsigned>(s->input[4]);
  if(round==0)*ptr<unsigned>(s->epoch_count)=observed;
  unsigned expected=*ptr<unsigned>(s->epoch_count);s->native_count=expected;
  if(observed!=expected){if(!failed(s))printf("COUNT_GUARD device_epoch frame=%u round=%u actual=%u expected=%u\n",s->frame,round,observed,expected);reject(s,815);}
  if(expected>=16384)reject(s,815);
  for(unsigned i=0;i<4;++i)if(expected>s->input_capacity[i])reject(s,816);
  if(h[0]!=0x53555246u||h[1]!=s->frame||h[2]!=round||h[3]!=1)reject(s,806);
 }
 __syncthreads();if(failed(s))return;
 unsigned n=s->native_count;uint4* original=ptr<uint4>(s->input[0]);
 guards(s);native_bindings(s,round);__syncthreads();if(failed(s))return;
 for(unsigned i=threadIdx.x;i<s->tets*4;i+=blockDim.x){
  unsigned value=ptr<unsigned>(s->input[5])[i];
  if(value>=s->nodes||value!=ptr<unsigned>(s->fixed_topology)[i])reject(s,827);
 }
 for(unsigned i=threadIdx.x;i<n;i+=blockDim.x){
  uint4 a=original[i];
  if((a.z>>20)!=0||(a.z&0xfffffu)>=s->tets)reject(s,829);
  if(round==0){ptr<uint4>(s->frame_info)[i]=a;ptr<unsigned>(s->frame_ids)[i]=ptr<unsigned>(s->input[1])[i];}
  else {
   uint4 b=ptr<uint4>(s->frame_info)[i];
   if(a.x!=b.x||a.y!=b.y||a.z!=b.z||(a.w&0x7fffffffu)!=(b.w&0x7fffffffu)||ptr<unsigned>(s->input[1])[i]!=ptr<unsigned>(s->frame_ids)[i])reject(s,830);
  }
 }
 __syncthreads();
 if(failed(s))return;
 if(COMPONENT_DENSE_POLICY)dense_after_valid_template(s,s->frame,round);else parallel_surface_plan(s,s->frame,round);
 __syncthreads();
 if(plan[0]){if(threadIdx.x==0)reject(s,30000+plan[0]);return;}
 device_robot_surface_copy(plan,original,ptr<unsigned>(s->input[1]),ptr<float>(s->input[2]),ptr<float>(s->input[3]),ptr<float4>(s->zero),ptr<uint4>(s->info),ptr<unsigned>(s->sid),ptr<float>(s->constraints),ptr<float>(s->lambda),ptr<float4>(s->rigid),ptr<unsigned>(s->count));
 __syncthreads();
 for(unsigned base=0;base<s->boundary_count;base+=blockDim.x){
  device_robot_surface_append(plan,ptr<uint4>(s->selected),original,ptr<float>(s->input[2]),ptr<float4>(s->input[6]),ptr<float4>(s->input[7]),s->plane,0,ptr<uint4>(s->info),ptr<float>(s->constraints),base);
 }
 __syncthreads();
 for(unsigned base=0;base<s->boundary_count;base+=blockDim.x){
  device_robot_surface_history_restore(plan,ptr<uint4>(s->selected),h,ptr<float>(s->cache),ptr<unsigned>(s->active),ptr<float>(s->lambda),ptr<uint4>(s->info),base);
 }
 __syncthreads();component_record(s,0);__syncthreads();guards(s);__syncthreads();
 if(threadIdx.x==0){
  if(failed(s))*ptr<unsigned>(s->count)=0;
  else {ptr<unsigned long long>(s->telemetry)[0]++;ptr<unsigned long long>(s->telemetry)[2]+=plan[2]-plan[1];}
 }
}
extern "C" __global__ void surface_stream_after_query(SurfaceStreamState* s){
 guards(s);__syncthreads();if(failed(s))return;
 if(ptr<unsigned>(s->header)[3]!=2){if(threadIdx.x==0)reject(s,819);return;}
 component_record(s,1);__syncthreads();
 for(unsigned i=threadIdx.x;i<s->native_count;i+=blockDim.x)ptr<uint4>(s->input[0])[i]=ptr<uint4>(s->info)[i];
}
extern "C" __global__ void surface_stream_bind_reaction(SurfaceStreamState* s,uint64_t source,unsigned capacity){
 guards(s);__syncthreads();if(failed(s))return;
 if(ptr<unsigned>(s->header)[3]!=2){if(threadIdx.x==0)reject(s,820);return;}
 unsigned n=s->native_count,nc=ptr<unsigned>(s->plan)[2];
 if(n>capacity){if(threadIdx.x==0)reject(s,820);return;}
 if(threadIdx.x==0)s->input[8]=source;
 for(unsigned i=threadIdx.x;i<n;i+=blockDim.x){
  ptr<float4>(s->rigid)[i]=ptr<float4>(source)[i];
  ptr<float4>(s->rigid)[nc+i]=ptr<float4>(source)[n+i];
 }
}
extern "C" __global__ void surface_stream_publish(SurfaceStreamState* s){
 unsigned* h=ptr<unsigned>(s->header);unsigned* plan=ptr<unsigned>(s->plan);
 guards(s);__syncthreads();if(failed(s))return;
 unsigned n=plan[1],nc=plan[2];
 for(unsigned i=n+threadIdx.x;i<nc;i+=blockDim.x){
  float4 a=ptr<float4>(s->rigid)[i],b=ptr<float4>(s->rigid)[nc+i];
  if(a.x!=0||a.y!=0||a.z!=0||a.w!=0||b.x!=0||b.y!=0||b.z!=0||b.w!=0)reject(s,823);
 }
 __syncthreads();if(failed(s))return;
 component_record(s,2);__syncthreads();
 device_robot_surface_history_store(plan,ptr<uint4>(s->selected),h,ptr<float>(s->lambda),ptr<uint4>(s->info),ptr<float>(s->cache),ptr<unsigned>(s->seen),ptr<unsigned>(s->active));
 __syncthreads();
 if(h[6]){if(threadIdx.x==0)reject(s,20000+h[6]);return;}
 for(unsigned i=threadIdx.x;i<n;i+=blockDim.x){
  ptr<uint4>(s->input[0])[i]=ptr<uint4>(s->info)[i];
  ptr<float>(s->input[3])[i]=ptr<float>(s->lambda)[i];
  ptr<float4>(s->input[8])[i]=ptr<float4>(s->rigid)[i];
  ptr<float4>(s->input[8])[n+i]=ptr<float4>(s->rigid)[nc+i];
 }
 __syncthreads();guards(s);__syncthreads();
 if(threadIdx.x==0&&!failed(s))ptr<unsigned long long>(s->telemetry)[1]++;
}
