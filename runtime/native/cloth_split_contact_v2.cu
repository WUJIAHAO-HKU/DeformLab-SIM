// Owned-buffer candidate: single-cloth VT/EE split contact with FP64 geometry.
// The table has the pinned 400-byte PxgFEMCloth layout. Host binding must verify it.
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
extern "C" {
__device__ unsigned long long cloth_split_counts[10];
}
struct V { double x,y,z; };
__device__ V operator+(V a,V b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
__device__ V operator-(V a,V b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
__device__ V operator*(V a,double s){return {a.x*s,a.y*s,a.z*s};}
__device__ double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ V cross(V a,V b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
__device__ V vec(float4 a){return {double(a.x),double(a.y),double(a.z)};}
__device__ double clamp01(double a){return fmin(1.,fmax(0.,a));}
template<class T> __device__ T field(const unsigned char* table,unsigned offset){return *reinterpret_cast<const T*>(table+offset);}
__device__ bool finite(V a){return isfinite(a.x)&&isfinite(a.y)&&isfinite(a.z);}
__device__ void reject(){atomicAdd(cloth_split_counts+6,1ull);}

// Return barycentric closest point; the edge candidates also handle degenerate triangles.
__device__ void point_triangle(V p,V a,V b,V c,double* w,V& gap,V& fallback){
 V e=b-a,f=c-a,r=p-a;double ee=dot(e,e),ff=dot(f,f),ef=dot(e,f);
 double det=ee*ff-ef*ef,best=INFINITY;
 fallback=cross(e,f);
 if(det>1e-28*fmax(ee*ff,1e-300)){
  double u=(ff*dot(r,e)-ef*dot(r,f))/det,v=(ee*dot(r,f)-ef*dot(r,e))/det;
  if(u>=0&&v>=0&&u+v<=1){gap=r-e*u-f*v;best=dot(gap,gap);w[0]=1-u-v;w[1]=u;w[2]=v;}
 }
 V q[3]={a,b,c};
 for(int i=0;i<3;++i){int j=(i+1)%3;V edge=q[j]-q[i];double l=dot(edge,edge);
  double t=l>1e-30?clamp01(dot(p-q[i],edge)/l):0.;V g=(p-q[i])-edge*t;double d=dot(g,g);
  if(d<best){best=d;gap=g;w[0]=w[1]=w[2]=0;w[i]=1-t;w[j]=t;}
 }
}
__device__ void segment_segment(V a,V b,V c,V d,double& s,double& t,V& gap,V& fallback){
 V e=b-a,f=d-c,r=a-c;double aa=dot(e,e),bb=dot(e,f),cc=dot(f,f),dd=dot(e,r),ee=dot(f,r);
 fallback=cross(e,f);s=t=0;
 if(aa<=1e-30&&cc<=1e-30){gap=r;return;}
 if(aa<=1e-30)t=clamp01(ee/cc);
 else if(cc<=1e-30)s=clamp01(-dd/aa);
 else{
  double det=aa*cc-bb*bb;
  s=det>1e-14*aa*cc?clamp01((bb*ee-cc*dd)/det):0.;
  t=(bb*s+ee)/cc;
  if(t<0){t=0;s=clamp01(-dd/aa);}
  else if(t>1){t=1;s=clamp01((bb-dd)/aa);}
 }
 gap=r+e*s-f*t;
}

template<bool Edge,bool Friction,bool Query=false>
__device__ void solve(unsigned char* table,const uint4* rows,const unsigned* count,float h){
 if(blockIdx.x==0&&threadIdx.x==0)atomicAdd(cloth_split_counts+(Query?(8+(Edge?1:0)):((Edge?2:0)+(Friction?1:0))),1ull);
 if(!(h>0)||!isfinite(h)){if(blockIdx.x==0&&threadIdx.x==0)reject();return;}
 const unsigned n=field<unsigned>(table,316);
 const unsigned nt=field<unsigned>(table,320);
 float4* p=field<float4*>(table,16);float4* v=field<float4*>(table,8);
 float4* pending=field<float4*>(table,248);const uint4* triangles=field<uint4*>(table,72);
 const float* friction=field<float*>(table,48);
 const double rest=field<float>(table,384);
 if(!Query&&!Friction){
  float4* saved=field<float4*>(table,280); // Private slot of candidate-owned descriptor only.
  if(saved)for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)saved[i]=make_float4(0,0,0,pending[i].w);
 }

 for(unsigned row=blockIdx.x*blockDim.x+threadIdx.x;row<*count;row+=gridDim.x*blockDim.x){
  uint4 record=rows[row];if(!(record.w&(1u<<29)))continue;
  atomicAdd(cloth_split_counts+4,1ull);
  if((record.x>>20)!=0||(record.z>>20)!=0){reject();continue;}
  unsigned first=record.x&0xfffff,second=record.z&0xfffff;
  if(second>=nt||(Edge?first>=nt:first>=n)){reject();continue;}
  unsigned ids[4];uint4 tb=triangles[second];unsigned tbv[3]={tb.x,tb.y,tb.z};
  if(Edge){
   unsigned ea=record.w&16383,eb=(record.w>>14)&32767;
   if(ea>2||eb>2){reject();continue;}
   uint4 ta=triangles[first];unsigned tav[3]={ta.x,ta.y,ta.z};
   ids[0]=tav[ea];ids[1]=tav[(ea+1)%3];ids[2]=tbv[eb];ids[3]=tbv[(eb+1)%3];
  }else{ids[0]=first;ids[1]=tb.x;ids[2]=tb.y;ids[3]=tb.z;}
  bool valid=true;for(int i=0;i<4;++i)valid&=ids[i]<n;
  if(!valid){reject();continue;}
  V x[4];double inverse[4],degree[4],weights[4];
  for(int i=0;i<4;++i){float4 pi=p[ids[i]];x[i]=vec(pi);inverse[i]=pi.w;degree[i]=pending[ids[i]].w;
   valid&=finite(x[i])&&isfinite(inverse[i])&&inverse[i]>=0&&isfinite(degree[i])&&degree[i]>=0;}
  if(!valid||!(rest>=0)||!isfinite(rest)){reject();continue;}
  // Translate in FP64, preserving the exact differences between input FP32 coordinates.
  V origin=x[0];for(int i=0;i<4;++i)x[i]=x[i]-origin;
  V gap,fallback;
  if(Edge){double s,t;segment_segment(x[0],x[1],x[2],x[3],s,t,gap,fallback);
   weights[0]=1-s;weights[1]=s;weights[2]=-(1-t);weights[3]=-t;
  }else{double bary[3];point_triangle(x[0],x[1],x[2],x[3],bary,gap,fallback);
   weights[0]=1;for(int i=0;i<3;++i)weights[i+1]=-bary[i];}
  double distance=sqrt(dot(gap,gap));V normal;
  if(distance>1e-14)normal=gap*(1./distance);
  else{double fl=sqrt(dot(fallback,fallback));if(!(fl>1e-14)){reject();continue;}normal=fallback*(1./fl);}
  if(Query){
   if(distance<2*rest)for(int i=0;i<4;++i)if(weights[i]!=0)atomicAdd(&pending[ids[i]].w,float(fabs(weights[i])));
   continue;
  }
  // Weighted Cauchy majorizer: degree_i=sum_c |w_ci|, including every nonzero contribution.
  double den=0;for(int i=0;i<4;++i)den+=fabs(weights[i])*inverse[i]*degree[i];
  if(!(den>1e-14))continue;
  double lambda=fmax(0.,2*rest-distance)/den;if(lambda==0)continue;
  V correction=normal*lambda;
  if(Friction){
   V relative={0,0,0};double mu=0;
   for(int i=0;i<4;++i){V vi=vec(v[ids[i]]);double m=friction[ids[i]];
    valid&=finite(vi)&&isfinite(m)&&m>=0;relative=relative+vi*weights[i];mu+=m*.25;}
   if(!valid){reject();continue;}
   V tangent=relative-normal*dot(relative,normal);double speed=sqrt(dot(tangent,tangent));
   if(!(speed>1e-14)||mu==0)continue;
   correction=tangent*(-fmin(double(h)*speed/den,mu*lambda)/speed);
  }
  atomicAdd(cloth_split_counts+5,1ull);
  for(int i=0;i<4;++i){double factor=inverse[i]*degree[i]*weights[i];if(factor==0)continue;
   V dx=correction*factor;atomicAdd(&pending[ids[i]].x,float(dx.x));atomicAdd(&pending[ids[i]].y,float(dx.y));atomicAdd(&pending[ids[i]].z,float(dx.z));
   if(dx.x!=0||dx.y!=0||dx.z!=0)atomicAdd(cloth_split_counts+7,1ull);
  }
 }
}
extern "C" __global__ void cloth_split_vt_normal(unsigned char* t,const uint4* r,const unsigned* n,float h){solve<false,false>(t,r,n,h);}
extern "C" __global__ void cloth_split_vt_friction(unsigned char* t,const uint4* r,const unsigned* n,float h){solve<false,true>(t,r,n,h);}
extern "C" __global__ void cloth_split_ee_normal(unsigned char* t,const uint4* r,const unsigned* n,float h){solve<true,false>(t,r,n,h);}
extern "C" __global__ void cloth_split_ee_friction(unsigned char* t,const uint4* r,const unsigned* n,float h){solve<true,true>(t,r,n,h);}

extern "C" __global__ void cloth_split_vt_query(unsigned char* t,const uint4* r,const unsigned* n,float h){solve<false,false,true>(t,r,n,h);}
extern "C" __global__ void cloth_split_ee_query(unsigned char* t,const uint4* r,const unsigned* n,float h){solve<true,false,true>(t,r,n,h);}
