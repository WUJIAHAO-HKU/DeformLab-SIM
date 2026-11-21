// Owned-buffer general contact port. No PhysX pointers and no native state writes.
// Maps use mass-whitened scene velocity coordinates; kinematic motion is affine.
// This maps a supplied impulse; it does not solve Coulomb contact or manage history.
#include <cuda_runtime.h>
#include <math.h>
constexpr int CAP=96, IN=2+6+9+6*CAP+CAP+3, OUT=3*CAP+3+3*CAP+6+1+9+3;
extern "C" __global__ void map_contact_port(const double* input,double* output,int* status,int cases) {
 int sample=blockIdx.x*blockDim.x+threadIdx.x;if(sample>=cases)return;
 const double* x=input+sample*IN;double* y=output+sample*OUT;
 // Validate every input before publishing any output.
 for(int i=0;i<IN;++i){if(!isfinite(x[i])){status[sample]=1;return;}
  if(fabs(x[i])>1e50){status[sample]=4;return;}}
 if(x[0]<1||x[0]>CAP||x[1]<1||x[1]>3){status[sample]=2;return;}
 int n=int(x[0]),k=int(x[1]);
 if(n<1||n>CAP||x[0]!=n||k<1||k>3||x[1]!=k){status[sample]=2;return;}
 const double* oa=x+2;const double* ob=oa+3;const double* T=ob+3;
 const double* A=T+9;const double* B=A+3*CAP;const double* u=B+3*CAP;const double* lambda=u+CAP;
 for(int i=0;i<k;++i)for(int j=0;j<k;++j){
  double v=0;for(int a=0;a<3;++a)v+=T[3*i+a]*T[3*j+a];
  if(fabs(v-(i==j?1.:0.))>1e-12){status[sample]=3;return;}
 }
 for(int i=0;i<OUT;++i)y[i]=0.;
 double* C=y;double* relative=C+3*CAP;double* qa=relative+3;double* qb=qa+CAP;double* total=qb+CAP;
 double* world_a=total+CAP;double* world_b=world_a+3;double* affine=world_b+3;double* W=affine+1;double* after=W+9;
 double p[3]={},point[3]={};
 for(int a=0;a<3;++a){
  for(int i=0;i<k;++i)p[a]+=T[3*i+a]*lambda[i];
  point[a]=ob[a]-oa[a];
  for(int j=0;j<n;++j)point[a]+=(B[a*CAP+j]-A[a*CAP+j])*u[j];
  world_a[a]=-p[a];world_b[a]=p[a];*affine+=p[a]*(ob[a]-oa[a]);
 }
 for(int j=0;j<n;++j){
  for(int a=0;a<3;++a){qa[j]-=A[a*CAP+j]*p[a];qb[j]+=B[a*CAP+j]*p[a];}
  total[j]=qa[j]+qb[j];
  for(int i=0;i<k;++i)for(int a=0;a<3;++a)C[i*CAP+j]+=T[3*i+a]*(B[a*CAP+j]-A[a*CAP+j]);
 }
 for(int i=0;i<k;++i){
  for(int a=0;a<3;++a)relative[i]+=T[3*i+a]*point[a];
  after[i]=relative[i];
  for(int j=0;j<n;++j)after[i]+=C[i*CAP+j]*total[j];
  for(int h=0;h<k;++h)for(int j=0;j<n;++j)W[i*3+h]+=C[i*CAP+j]*C[h*CAP+j];
 }
 status[sample]=0;
}

