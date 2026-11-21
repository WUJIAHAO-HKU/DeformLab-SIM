// Experimental owned-buffer GPU feasibility guard. Not a live PhysX backend.
// Compact active rows; parallel node/plane projection. Active append order may
// vary; no bitwise determinism claim. Not the minimum-change CPU QP.
// Each row contains the origin, so each exact local projection is passive.
// Double scratch; native-format FP32 velocity published only after certificates.
#include <cuda_runtime.h>
#include <math.h>
using U = unsigned long long;
__device__ double3 add(double3 a,double3 b){return make_double3(a.x+b.x,a.y+b.y,a.z+b.z);}
__device__ double3 sub(double3 a,double3 b){return make_double3(a.x-b.x,a.y-b.y,a.z-b.z);}
__device__ double3 mul(double3 a,double s){return make_double3(a.x*s,a.y*s,a.z*s);}
__device__ double dot(double3 a,double3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ double3 cross(double3 a,double3 b){return make_double3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
__device__ double3 xyz(float4 a){return make_double3(a.x,a.y,a.z);}
__device__ bool finite3(double3 a){return isfinite(a.x)&&isfinite(a.y)&&isfinite(a.z);}
__device__ bool finite4(float4 a){return finite3(xyz(a))&&isfinite(a.w);}
__device__ unsigned node(uint4 t,int i){return i==0?t.x:i==1?t.y:i==2?t.z:t.w;}
__device__ double component(double3 v,int i){return i==0?v.x:i==1?v.y:v.z;}
__device__ void symmap(const double3* v,const double3* g,double h,double a[3][3]){
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){
    double z=0;for(int k=0;k<4;++k)z+=component(v[k],i)*component(g[k],j)+component(v[k],j)*component(g[k],i);
    a[i][j]=(i==j?1.:0.)+.5*h*z;
  }
}
__device__ double lowest(double a[3][3],double3& n){
  double q[3][3]={{1,0,0},{0,1,0},{0,0,1}};
  for(int iter=0;iter<64;++iter){
    int p=0,r=1;if(fabs(a[0][2])>fabs(a[p][r])){p=0;r=2;}if(fabs(a[1][2])>fabs(a[p][r])){p=1;r=2;}
    double scale=fmax(1.,fmax(fabs(a[0][0]),fmax(fabs(a[1][1]),fabs(a[2][2]))));
    if(fabs(a[p][r])<1e-15*scale)break;
    double tau=(a[r][r]-a[p][p])/(2*a[p][r]);
    double t=copysign(1.,tau)/(fabs(tau)+hypot(1.,tau)),c=1./sqrt(1+t*t),s=t*c,apr=a[p][r];
    a[p][p]-=t*apr;a[r][r]+=t*apr;a[p][r]=a[r][p]=0;
    for(int k=0;k<3;++k){
      if(k!=p&&k!=r){double u=a[k][p],v=a[k][r];a[k][p]=a[p][k]=c*u-s*v;a[k][r]=a[r][k]=s*u+c*v;}
      double u=q[k][p],v=q[k][r];q[k][p]=c*u-s*v;q[k][r]=s*u+c*v;
    }
  }
  int k=a[1][1]<a[0][0]?1:0;if(a[2][2]<a[k][k])k=2;
  n=make_double3(q[0][k],q[1][k],q[2][k]);return a[k][k];
}
// stats: invalid bitset, iterations, volume projections, plane projections,
// converged, current violated tet count, published nodes, reserved, reserved, init.
extern "C" __global__ void cone_init(const float4* p,const float4* v,unsigned n,double h,double eta,
 double3* w,double3* internal,double3* floor_impulse,double* energy,U* stats){
  if(blockIdx.x||threadIdx.x)return;
  stats[9]=1;
  if(!n||n>100000||!isfinite(h)||h<=0||!isfinite(eta)||eta<=0||eta>=1){stats[0]|=1;return;}
  double e=0;
  for(unsigned i=0;i<n;++i){
    if(!finite4(p[i])||p[i].w<=0||!finite4(v[i])){stats[0]|=2;return;}
    w[i]=xyz(v[i]);internal[i]=floor_impulse[i]=make_double3(0,0,0);e+=dot(w[i],w[i])/p[i].w;
  }
  if(!isfinite(e)){stats[0]|=2;return;}energy[0]=e;
}
extern "C" __global__ void cone_prepare(const float4* p,unsigned n,const uint4* t,unsigned nt,
 const double* restdet,double h,double eta,const double* energy,double3* gradients,double* target,U* stats){
  unsigned j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=nt||stats[0])return;
  uint4 ids=t[j];double3 x[4];
  for(int k=0;k<4;++k){unsigned i=node(ids,k);if(i>=n){atomicOr(stats,4ull);return;}x[k]=xyz(p[i]);}
  double3 u=sub(x[1],x[0]),v=sub(x[2],x[0]),w=sub(x[3],x[0]);double d=dot(u,cross(v,w));
  if(!isfinite(d)||d==0||!isfinite(restdet[j])||restdet[j]==0||d/restdet[j]<=0){atomicOr(stats,8ull);return;}
  double3 g[4];g[1]=mul(cross(v,w),1/d);g[2]=mul(cross(w,u),1/d);g[3]=mul(cross(u,v),1/d);g[0]=mul(add(add(g[1],g[2]),g[3]),-1);
  double margin=1e-9;
  for(int k=0;k<4;++k){
    unsigned i=node(ids,k);double bound=sqrt(energy[0]*p[i].w),ev=sqrt(3.)*0x1p-24*bound;
    double ep=sqrt(3.)*0x1p-24*(fmax(fabs(x[k].x),fmax(fabs(x[k].y),fabs(x[k].z)))+h*(bound+ev));
    margin+=(h*ev+ep)*sqrt(dot(g[k],g[k]));gradients[4*j+k]=g[k];
  }
  target[j]=eta+margin;
  if(!isfinite(target[j])||target[j]>=1)atomicOr(stats,16ull);
}
extern "C" __global__ void cone_reset(U* stats){if(!blockIdx.x&&!threadIdx.x&&!stats[4])stats[5]=0;}
extern "C" __global__ void cone_find(const uint4* t,unsigned nt,const double3* w,
 const double3* gradients,const double* target,double h,double3* normals,unsigned* active,U* stats){
  unsigned j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=nt||stats[0]||stats[4])return;
  double3 v[4];for(int k=0;k<4;++k)v[k]=w[node(t[j],k)];double a[3][3];symmap(v,gradients+4*j,h,a);
  double3 normal;double e=lowest(a,normal);normals[j]=make_double3(0,0,0);
  if(!isfinite(e)||!finite3(normal)){atomicOr(stats,32ull);return;}
  if(e<target[j]-1e-9){normals[j]=normal;unsigned slot=unsigned(atomicAdd(stats+5,1ull));active[slot]=j;}
}
extern "C" __global__ void cone_apply(const float4* p,unsigned n,const uint4* t,unsigned nt,
 const double3* gradients,const double* target,const double3* normals,const unsigned* active,double h,int plane,double floor,
 double3* w,double3* internal,double3* floor_impulse,U* stats){
  if(blockIdx.x||stats[0]||stats[4])return;
  __shared__ U changes;
  if(threadIdx.x==0){
    changes=0;++stats[1];
    if((plane!=0&&plane!=1)||(plane&&!isfinite(floor)))stats[0]|=2048;
    for(U c=0;c<stats[5]&&!stats[0];++c){
      unsigned j=active[c];
      if(j>=nt){stats[0]|=4096;break;}
      double3 normal=normals[j];double3 a[4];double value=0,denom=0;
      for(int k=0;k<4;++k){unsigned i=node(t[j],k);a[k]=mul(normal,h*dot(gradients[4*j+k],normal));value+=dot(a[k],w[i]);denom+=p[i].w*dot(a[k],a[k]);}
      double deficit=target[j]-1-value;if(deficit<=0)continue;
      if(!isfinite(denom)||denom<=0){stats[0]|=32;break;}
      double lambda=deficit/denom;
      for(int k=0;k<4;++k){unsigned i=node(t[j],k);double3 impulse=mul(a[k],lambda);internal[i]=add(internal[i],impulse);w[i]=add(w[i],mul(impulse,p[i].w));}
      ++stats[2];
    }
  }
  __syncthreads();
  if(stats[0])return;
  U local=0;
  if(plane){
    for(unsigned i=threadIdx.x;i<n;i+=blockDim.x){
      double lower=-fmax(double(p[i].z)-floor,0.)/h;
      if(w[i].z<lower){floor_impulse[i].z+=(lower-w[i].z)/p[i].w;w[i].z=lower;++local;}
    }
  }
  atomicAdd(&changes,local);
  __syncthreads();
  if(threadIdx.x==0){stats[3]+=changes;if(!stats[5]&&!changes)stats[4]=1;}

}
extern "C" __global__ void cone_verify_nodes(const float4* p,const float4* v,unsigned n,
 const double3* w,double h,int plane,double floor,double* energy,U* stats){
  if(blockIdx.x||threadIdx.x||stats[0])return;
  if(!stats[4]){stats[0]|=64;return;}
  double dk=0;
  for(unsigned i=0;i<n;++i){
    double3 z=xyz(make_float4(float(w[i].x),float(w[i].y),float(w[i].z),0));
    if(!finite3(z)){stats[0]|=128;return;}
    float future=fmaf(float(h),float(z.z),p[i].z);
    if(plane&&future<fmin(double(p[i].z),floor)){stats[0]|=256;return;}
    double3 old=xyz(v[i]);dk+=(dot(z,z)-dot(old,old))/(2*p[i].w);
  }
  energy[1]=dk;if(dk>1e-12||!isfinite(dk))stats[0]|=512;
}
extern "C" __global__ void cone_verify_tets(const float4* p,const uint4* t,unsigned nt,const double3* w,
 const double3* gradients,const double* restdet,double h,double eta,U* stats){
  unsigned j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=nt||stats[0])return;
  double3 v[4],future[4],delta[4];
  for(int k=0;k<4;++k){unsigned i=node(t[j],k);float4 vel=make_float4(float(w[i].x),float(w[i].y),float(w[i].z),0);
    v[k]=xyz(vel);future[k]=make_double3(fmaf(float(h),vel.x,p[i].x),fmaf(float(h),vel.y,p[i].y),fmaf(float(h),vel.z,p[i].z));delta[k]=sub(future[k],xyz(p[i]));}
  double a[3][3];double3 normal;symmap(v,gradients+4*j,h,a);double ev=lowest(a,normal);
  symmap(delta,gradients+4*j,1.,a);double ex=lowest(a,normal);
  double J=dot(sub(future[1],future[0]),cross(sub(future[2],future[0]),sub(future[3],future[0])))/restdet[j];
  if(!isfinite(ev)||!isfinite(ex)||!isfinite(J)||ev<eta-1e-8||ex<eta-1e-8||J<=0)atomicOr(stats,1024ull);
}
extern "C" __global__ void cone_publish(float4* v,unsigned n,const double3* w,U* stats){
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||stats[0]||!stats[4])return;
  v[i].x=float(w[i].x);v[i].y=float(w[i].y);v[i].z=float(w[i].z);atomicAdd(stats+6,1ull);
}
