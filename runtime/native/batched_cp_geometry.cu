// Read-only root tetrahedron geometry observation. One block owns each body.
#include <cuda_runtime.h>
#include <math.h>
struct BGBody {const float4* p;const uint4* t;const double* reference;float4* previous;unsigned n,nt;};
struct BGStats {unsigned long long calls,tet_checks,invalid_tets,nonfinite_tets,first_step,first_cycle;double minimum_j;unsigned critical_tet,padding;float4 before[4],after[4];};
static_assert(sizeof(BGBody)==40,"Geometry descriptor ABI");
static_assert(sizeof(BGStats)==192,"Geometry statistics ABI");
extern "C" __global__ void batched_geometry_check(const BGBody* bodies,unsigned count,unsigned stage,unsigned stages,unsigned long long step,unsigned long long cycle,BGStats* stats){
 unsigned body=blockIdx.y,lane=threadIdx.x;
 if(blockIdx.x||body>=count||stage>=stages||blockDim.x!=128)return;
 const BGBody b=bodies[body];BGStats& row=stats[body*stages+stage];
 __shared__ double minima[128];__shared__ unsigned indices[128],invalid[128],nonfinite[128],nan_index[128];
 double minimum=INFINITY;unsigned index=~0u,bad=0,nans=0,first_nan=~0u;
 for(unsigned k=lane;k<b.nt;k+=128){
  uint4 ids=b.t[k];float4 a=b.p[ids.x],c=b.p[ids.y],d=b.p[ids.z],e=b.p[ids.w];
  double ux=double(c.x)-a.x,uy=double(c.y)-a.y,uz=double(c.z)-a.z;
  double vx=double(d.x)-a.x,vy=double(d.y)-a.y,vz=double(d.z)-a.z;
  double wx=double(e.x)-a.x,wy=double(e.y)-a.y,wz=double(e.z)-a.z;
  double ratio=(ux*(vy*wz-vz*wy)+uy*(vz*wx-vx*wz)+uz*(vx*wy-vy*wx))/b.reference[k];
  if(!isfinite(ratio)){++nans;first_nan=min(first_nan,k);}
  else{if(ratio<=0)++bad;if(ratio<minimum||(ratio==minimum&&k<index)){minimum=ratio;index=k;}}
 }
 minima[lane]=minimum;indices[lane]=index;invalid[lane]=bad;nonfinite[lane]=nans;nan_index[lane]=first_nan;__syncthreads();
 for(unsigned stride=64;stride;stride/=2){
  if(lane<stride){unsigned other=lane+stride;
   if(minima[other]<minima[lane]||(minima[other]==minima[lane]&&indices[other]<indices[lane])){minima[lane]=minima[other];indices[lane]=indices[other];}
   invalid[lane]+=invalid[other];nonfinite[lane]+=nonfinite[other];nan_index[lane]=min(nan_index[lane],nan_index[other]);
  }__syncthreads();
 }
 if(!lane){
  if(!row.calls||minima[0]<row.minimum_j)row.minimum_j=minima[0];
  ++row.calls;row.tet_checks+=b.nt;row.invalid_tets+=invalid[0];row.nonfinite_tets+=nonfinite[0];
  if(!row.first_step&&(invalid[0]||nonfinite[0])){
   unsigned k=nonfinite[0]?nan_index[0]:indices[0];row.first_step=step;row.first_cycle=cycle;row.critical_tet=k;
   uint4 ids=b.t[k];unsigned vertices[]={ids.x,ids.y,ids.z,ids.w};
   for(unsigned i=0;i<4;++i){row.before[i]=b.previous[vertices[i]];row.after[i]=b.p[vertices[i]];}
  }
 }__syncthreads();
 // Copy only after every lane has completed all reads of the preceding stage.
 for(unsigned i=lane;i<b.n;i+=128)b.previous[i]=b.p[i];
}
