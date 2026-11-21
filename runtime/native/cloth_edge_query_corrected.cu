// Explicit single-cloth EE validity correction; no integration or generator change.
#include <cuda_runtime.h>
#include <math.h>
extern "C" { __device__ unsigned long long cloth_ee_query_counts[4]; }
struct EQV {double x,y,z;};
__device__ EQV eq_sub(float4 a,float4 b){return {double(a.x)-b.x,double(a.y)-b.y,double(a.z)-b.z};}
__device__ double eq_dot(EQV a,EQV b){return a.x*b.x+a.y*b.y+a.z*b.z;}
template<class T> __device__ T eq_field(const unsigned char* t,unsigned off){return *reinterpret_cast<const T*>(t+off);}
extern "C" __global__ void cloth_ee_query_corrected(unsigned char* table,uint4* rows,const unsigned* count){
 if(!blockIdx.x&&!threadIdx.x)atomicAdd(cloth_ee_query_counts,1ull);
 const unsigned n=eq_field<unsigned>(table,316),nt=eq_field<unsigned>(table,320);
 const float4* positions=eq_field<const float4*>(table,16);const uint4* triangles=eq_field<const uint4*>(table,72);
 float4* pending=eq_field<float4*>(table,248);double separation=2.*eq_field<float>(table,384);
 for(unsigned k=blockIdx.x*blockDim.x+threadIdx.x;k<*count;k+=gridDim.x*blockDim.x){
  uint4 row=rows[k];rows[k].w=row.w&~(1u<<29);
  if(!(row.w&(1u<<31)))continue;
  atomicAdd(cloth_ee_query_counts+1,1ull);
  unsigned a=row.x&0xfffff,b=row.z&0xfffff,ea=row.w&16383,eb=(row.w>>14)&32767;
  if((row.x>>20)||(row.z>>20)||a>=nt||b>=nt||ea>2||eb>2||!(separation>=0)||!isfinite(separation)){
   atomicAdd(cloth_ee_query_counts+3,1ull);continue;
  }
  uint4 ta=triangles[a],tb=triangles[b];unsigned aa[3]={ta.x,ta.y,ta.z},bb[3]={tb.x,tb.y,tb.z};
  unsigned ids[4]={aa[ea],aa[(ea+1)%3],bb[eb],bb[(eb+1)%3]};
  bool valid=true;for(int i=0;i<4;++i)valid&=ids[i]<n;
  if(!valid){atomicAdd(cloth_ee_query_counts+3,1ull);continue;}
  float4 p[4];for(int i=0;i<4;++i){p[i]=positions[ids[i]];valid&=isfinite(p[i].x)&&isfinite(p[i].y)&&isfinite(p[i].z);}
  if(!valid){atomicAdd(cloth_ee_query_counts+3,1ull);continue;}
  EQV e=eq_sub(p[1],p[0]),f=eq_sub(p[3],p[2]),r=eq_sub(p[0],p[2]);
  double A=eq_dot(e,e),B=eq_dot(e,f),C=eq_dot(f,f),D=eq_dot(e,r),E=eq_dot(f,r);
  // Preserve native degenerate/parallel and interior-feature cutoffs. Fix only
  // the unequal-length line-parameter equations, evaluated in relative FP64.
  if(A<double(__uint_as_float(0x219392efu))||C<double(__uint_as_float(0x219392efu)))continue;
  double cosine=B/sqrt(A*C);
  if(fabs(cosine)>double(__uint_as_float(0x3f7ff972u)))continue;
  double den=A*C-B*B;if(!(den>0))continue;
  double s=(B*E-C*D)/den,t=(A*E-B*D)/den;
  const double low=__uint_as_float(0x358637bdu),high=__uint_as_float(0x3f7fffefu);
  if(s<low||s>high||t<low||t>high)continue;
  EQV gap={r.x+s*e.x-t*f.x,r.y+s*e.y-t*f.y,r.z+s*e.z-t*f.z};
  if(eq_dot(gap,gap)>separation*separation)continue;
  double weights[4]={fabs(1-s),fabs(s),fabs(1-t),fabs(t)};
  for(int i=0;i<4;++i)if(weights[i]>double(__uint_as_float(0x3a83126fu)))atomicAdd(&pending[ids[i]].w,float(weights[i]));
  rows[k].w=row.w|(1u<<29);atomicAdd(cloth_ee_query_counts+2,1ull);
 }
}
