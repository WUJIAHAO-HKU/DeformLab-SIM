// Conservative new-feature initialization only. Certifies separation under
// linear interpolation between owned frame roots and current positions.
// It is not a continuous-collision guarantee for the actual solver trajectory.
#include "cloth_ee_side_history_probe.cu"
#include <float.h>
__device__ double fh_triple(SV w,SV u,SV v){return dot(w,cross(u,v));}
__device__ int fh_birth_side(const float4* old,const float4* cur,double* bernstein){
 for(int j=0;j<4;++j)if(!finite4(old[j])||!finite4(cur[j]))return 0;
 SV u0=sub(xyz(old[1]),xyz(old[0])),v0=sub(xyz(old[3]),xyz(old[2])),w0=sub(xyz(old[0]),xyz(old[2]));
 SV u1=sub(xyz(cur[1]),xyz(cur[0])),v1=sub(xyz(cur[3]),xyz(cur[2])),w1=sub(xyz(cur[0]),xyz(cur[2]));
 double nn=dot(cross(u0,v0),cross(u0,v0));
 if(!(dot(u0,u0)>1e-20&&dot(v0,v0)>1e-20&&nn>1e-10*dot(u0,u0)*dot(v0,v0)))return 0;
 bernstein[0]=fh_triple(w0,u0,v0);
 bernstein[1]=(fh_triple(w1,u0,v0)+fh_triple(w0,u1,v0)+fh_triple(w0,u0,v1))/3.;
 bernstein[2]=(fh_triple(w0,u1,v1)+fh_triple(w1,u0,v1)+fh_triple(w1,u1,v0))/3.;
 bernstein[3]=fh_triple(w1,u1,v1);
 if(fabs(bernstein[0])/sqrt(nn)<=1e-12)return 0;
 double scale=sqrt(fmax(dot(u0,u0),dot(u1,u1))*fmax(dot(v0,v0),dot(v1,v1))*fmax(dot(w0,w0),dot(w1,w1)));
 double tolerance=128.*DBL_EPSILON*scale;
 int side=bernstein[0]>0?1:-1;
 for(int i=0;i<4;++i)if(!isfinite(bernstein[i])||side*bernstein[i]<=tolerance)return 0;
 return side;
}
extern "C" __global__ void cloth_ee_history_birth_probe(const float4* old,const float4* cur,int* side,double* coefficients){
 if(blockIdx.x||threadIdx.x)return;
 for(int i=0;i<4;++i)coefficients[i]=0;
 *side=fh_birth_side(old,cur,coefficients);
}
