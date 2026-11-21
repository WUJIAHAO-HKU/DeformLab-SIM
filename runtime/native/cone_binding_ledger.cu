// Additional explicit impulse/energy ledger for the owned cone guard.
// Native velocity-stage work is separate from guard work and floor reaction.
#include <cuda_runtime.h>
#include <math.h>
extern "C" __global__ void cone_ledger(const float4* p,const float4* incoming,
 const float4* before,const double3* working,const double3* internal,
 const double3* floor_impulse,unsigned n,double* out){
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  double m=1./p[i].w,x[3]={p[i].x,p[i].y,p[i].z};
  double v[3]={before[i].x,before[i].y,before[i].z};
  double old[3]={incoming[i].x,incoming[i].y,incoming[i].z};
  double w[3]={working[i].x,working[i].y,working[i].z};
  double a[3]={internal[i].x,internal[i].y,internal[i].z};
  double f[3]={floor_impulse[i].x,floor_impulse[i].y,floor_impulse[i].z};
  double guard=0,forcing=0,wi=0,wf=0,residual=0,published=0;
  for(int k=0;k<3;++k){
    atomicAdd(out+k,a[k]);atomicAdd(out+6+k,f[k]);
    int l=(k+1)%3,r=(k+2)%3;
    atomicAdd(out+3+k,x[l]*a[r]-x[r]*a[l]);
    atomicAdd(out+9+k,x[l]*f[r]-x[r]*f[l]);
    double vp=double(float(w[k]));
    guard+=.5*m*(w[k]*w[k]-v[k]*v[k]);
    published+=.5*m*(vp*vp-v[k]*v[k]);
    forcing+=.5*m*(v[k]*v[k]-old[k]*old[k]);
    wi+=a[k]*.5*(v[k]+w[k]);wf+=f[k]*.5*(v[k]+w[k]);
    residual+=fabs(m*(w[k]-v[k])-a[k]-f[k]);
  }
  atomicAdd(out+12,guard);atomicAdd(out+13,wi);atomicAdd(out+14,wf);
  atomicAdd(out+15,forcing);atomicAdd(out+16,residual);atomicAdd(out+17,published);
}
