#include "event_response_policy.h"
#include "event_material_reference.inc"
#ifndef EVENT_MODE
#error Explicit EVENT_MODE required
#endif
static_assert(EVENT_MODE>=0&&EVENT_MODE<=7,"event mode");
static __device__ float4 event_previous_position[65536];
static __device__ unsigned event_last_refresh;
static __device__ bool event_do_refresh;

static __device__ void event_decide(SurfaceStreamState* s,unsigned round){
 __shared__ double gaps[128],predicted[128],active_gaps[128],displacements[128],response[128],forces[128];
 __shared__ unsigned omitted[128],active[128];
 unsigned tid=threadIdx.x;
 if(s->nodes!=EVENT_EXPECTED_NODES||round>=256){if(tid==0)reject(s,843);return;}
 if(round==0&&tid==0)event_last_refresh=0;
 __syncthreads();
 const uint4* boundary=ptr<uint4>(s->boundary);const float4* x=ptr<float4>(s->input[6]);const unsigned* seen=ptr<unsigned>(s->seen);
 double mg=0,mp=0,ma=0,md=0,eta=0,force=0;unsigned no=0,na=0;
 for(unsigned j=tid;j<s->boundary_count;j+=blockDim.x){
  unsigned i=boundary[j].x;
  if(i>=s->nodes){reject(s,844);continue;}
  float4 a=x[i];if(!finite4(a)){reject(s,845);continue;}
  double dz=round?double(a.z)-event_previous_position[i].z:0.;event_previous_position[i]=a;
  if(fabs(double(a.x)-s->cx)>=s->hx||fabs(double(a.y)-s->cy)>=s->hy)continue;
  double g=fmax(0.,s->plane-double(a.z));md=fmax(md,fabs(dz));
  if(seen[i]){ma=fmax(ma,g);na++;}
  else{
   double forecast=fmax(0.,s->plane-double(a.z)+fmax(0.,-dz));
   mg=fmax(mg,g);mp=fmax(mp,forecast);no+=g>s->epsilon;
   eta+=event_material_gain[i]*forecast;
   force+=event_material_stiffness[i]*g;
  }
 }
 gaps[tid]=mg;predicted[tid]=mp;active_gaps[tid]=ma;displacements[tid]=md;response[tid]=eta;forces[tid]=force;omitted[tid]=no;active[tid]=na;
 __syncthreads();
 if(tid==0){
  mg=mp=ma=md=eta=force=0;no=na=0;
  for(unsigned k=0;k<blockDim.x;k++){
   mg=fmax(mg,gaps[k]);mp=fmax(mp,predicted[k]);ma=fmax(ma,active_gaps[k]);md=fmax(md,displacements[k]);eta+=response[k];force+=forces[k];no+=omitted[k];na+=active[k];
  }
  unsigned age=round-event_last_refresh;
  EventDecision d=response_policy(EVENT_MODE,round,age,mg,mp,eta,0.000005,24);
  event_do_refresh=d.refresh;if(d.refresh)event_last_refresh=round;
  auto t=component_trace+24*round;
  t[0]=s->frame;t[1]=round;t[2]=EVENT_MODE;t[3]=d.refresh;t[4]=d.reason;t[5]=no;t[6]=na;
  t[7]=__double_as_longlong(mg);t[8]=__double_as_longlong(mp);t[9]=__double_as_longlong(ma);
  t[12]=__double_as_longlong(eta);t[13]=__double_as_longlong(md);t[14]=age;t[15]=d.refresh?0:no;
  t[16]=__double_as_longlong(force);t[17]=s->nodes;t[18]=s->boundary_count;t[19]=__double_as_longlong(fmax(mg,ma));
  t[20]=__double_as_longlong(0.000005);t[23]=0x45565433;
 }
 __syncthreads();
}

// This is post-contact-row publication, which can precede further native work.
// Independent end-of-SimulationManager.step checks run on the actual published state.
static __device__ void event_post_contact(SurfaceStreamState* s){
 __shared__ double all_gap[128],missed_gap[128];
 unsigned tid=threadIdx.x;double a=0,b=0;
 const uint4* boundary=ptr<uint4>(s->boundary);const float4* x=ptr<float4>(s->input[6]);const unsigned* seen=ptr<unsigned>(s->seen);
 for(unsigned j=tid;j<s->boundary_count;j+=blockDim.x){
  unsigned i=boundary[j].x;float4 p=x[i];
  if(fabs(double(p.x)-s->cx)<s->hx&&fabs(double(p.y)-s->cy)<s->hy){
   double gap=fmax(0.,s->plane-double(p.z));a=fmax(a,gap);if(!seen[i])b=fmax(b,gap);
  }
 }
 all_gap[tid]=a;missed_gap[tid]=b;__syncthreads();
 if(tid==0){
  a=b=0;for(unsigned k=0;k<blockDim.x;k++){a=fmax(a,all_gap[k]);b=fmax(b,missed_gap[k]);}
  unsigned round=ptr<unsigned>(s->header)[2]-1;
  if(round<256){component_trace[24*round+21]=__double_as_longlong(a);component_trace[24*round+22]=__double_as_longlong(b);}else reject(s,846);
 }
 __syncthreads();
}
