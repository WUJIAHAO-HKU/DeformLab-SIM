// Bounded finite-segment separation screen for linear endpoint interpolation.
// FP64 margins are conservative numerical guards, not interval arithmetic CCD.
#include "cloth_ee_history_birth.cu"
extern "C" { __device__ unsigned long long cloth_history_finite_counts[5]; }
struct FBC {double lo,hi,b[4];SV n[3];unsigned depth;};
__device__ double fb_axis(SV v,int axis){return axis==0?v.x:axis==1?v.y:v.z;}
__device__ bool fb_same(const double* b,double tol){bool pos=true,neg=true;for(int i=0;i<4;++i){pos&=b[i]>tol;neg&=b[i]<-tol;}return pos||neg;}
__device__ __noinline__ int fh_finite_birth_side(const float4* old,const float4* cur,unsigned& kind,unsigned& visited,unsigned& boxes,unsigned max_depth=12,unsigned max_intervals=64){
 kind=visited=boxes=0;
 for(int i=0;i<4;++i)if(!finite4(old[i])||!finite4(cur[i]))return 0;
 SV o[4],c[4];for(int i=0;i<4;++i){o[i]=xyz(old[i]);c[i]=xyz(cur[i]);}
 SV u0=sub(o[1],o[0]),v0=sub(o[3],o[2]),w0=sub(o[0],o[2]);
 SV u1=sub(c[1],c[0]),v1=sub(c[3],c[2]),w1=sub(c[0],c[2]);
 SV n0=cross(u0,v0),n1=mul(add(cross(u1,v0),cross(u0,v1)),.5),n2=cross(u1,v1);
 double uu=fmax(dot(u0,u0),dot(u1,u1)),vv=fmax(dot(v0,v0),dot(v1,v1)),ww=fmax(dot(w0,w0),dot(w1,w1));
 double b[4]={fh_triple(w0,u0,v0),(fh_triple(w1,u0,v0)+fh_triple(w0,u1,v0)+fh_triple(w0,u0,v1))/3.,(fh_triple(w0,u1,v1)+fh_triple(w1,u0,v1)+fh_triple(w1,u1,v0))/3.,fh_triple(w1,u1,v1)};
 for(int i=0;i<4;++i)if(!isfinite(b[i]))return 0;
 if(dot(n0,n0)<=1e-10*dot(u0,u0)*dot(v0,v0)||fmin(dot(u0,u0),dot(v0,v0))<=1e-20)return 0;
 if(fabs(b[0])/sqrt(dot(n0,n0))<=1e-12||dot(n2,n2)==0||fabs(b[3])/sqrt(dot(n2,n2))<=1e-12)return 0;
 double tol=128.*DBL_EPSILON*sqrt(uu*vv*ww);int side=b[3]>0?1:-1;
 if(fb_same(b,tol)){kind=1;visited=1;return side;}
 kind=2;if(max_depth>12||max_intervals>64)return 0;
 double normal_tol=1e-5*sqrt(uu*vv),coordinate_scale=1e-12;
 for(int i=0;i<4;++i)for(int a=0;a<3;++a)coordinate_scale=fmax(coordinate_scale,fmax(fabs(fb_axis(o[i],a)),fabs(fb_axis(c[i],a))));
 double coordinate_tol=256.*DBL_EPSILON*coordinate_scale;
 FBC stack[14];unsigned size=1;stack[0].lo=0;stack[0].hi=1;stack[0].depth=0;
 for(int i=0;i<4;++i)stack[0].b[i]=b[i];stack[0].n[0]=n0;stack[0].n[1]=n1;stack[0].n[2]=n2;
 while(size){
  FBC cell=stack[--size];if(++visited>max_intervals)return 0;bool normal_safe=false;
  for(int a=0;a<3;++a){double lo=fb_axis(cell.n[0],a),hi=lo;for(int j=1;j<3;++j){lo=fmin(lo,fb_axis(cell.n[j],a));hi=fmax(hi,fb_axis(cell.n[j],a));}normal_safe|=lo>normal_tol||hi<-normal_tol;}
  if(normal_safe&&fb_same(cell.b,tol))continue;
  bool separated=false;
  for(int a=0;a<3;++a){
   double mn[2]={DBL_MAX,DBL_MAX},mx[2]={-DBL_MAX,-DBL_MAX};
   for(int i=0;i<4;++i){double x=fb_axis(o[i],a),dx=fb_axis(sub(c[i],o[i]),a),v=x+dx*cell.lo,w=x+dx*cell.hi;mn[i/2]=fmin(mn[i/2],fmin(v,w));mx[i/2]=fmax(mx[i/2],fmax(v,w));}
   separated|=mn[0]-mx[1]>coordinate_tol||mn[1]-mx[0]>coordinate_tol;
  }
  if(normal_safe&&separated){++boxes;continue;}
  if(normal_safe){
   SV midpoints[4];for(int i=0;i<4;++i)midpoints[i]=add(o[i],mul(sub(c[i],o[i]),(cell.lo+cell.hi)*.5));
   SV axis={0,0,0};double best=DBL_MAX;
   for(int j=0;j<4;++j){
    int a=j<2?2:0,b=a+1;SV e=sub(midpoints[b],midpoints[a]),w=sub(midpoints[j],midpoints[a]);double den=dot(e,e),t=den>0?fmin(1.,fmax(0.,dot(w,e)/den)):0;
    SV candidate=sub(w,mul(e,t));double dd=dot(candidate,candidate);if(dd<best){best=dd;axis=candidate;}
   }
   if(best>0&&best<DBL_MAX){
    double mn[2]={DBL_MAX,DBL_MAX},mx[2]={-DBL_MAX,-DBL_MAX};
    for(int i=0;i<4;++i){double x=dot(o[i],axis),dx=dot(sub(c[i],o[i]),axis),v=x+dx*cell.lo,w=x+dx*cell.hi;mn[i/2]=fmin(mn[i/2],fmin(v,w));mx[i/2]=fmax(mx[i/2],fmax(v,w));}
    if(mn[0]-mx[1]>coordinate_tol*sqrt(best)||mn[1]-mx[0]>coordinate_tol*sqrt(best)){++boxes;continue;}
   }
  }
  if(cell.depth==max_depth||size+2>14)return 0;
  double l[3],m[2];for(int i=0;i<3;++i)l[i]=(cell.b[i]+cell.b[i+1])*.5;for(int i=0;i<2;++i)m[i]=(l[i]+l[i+1])*.5;double midpoint=(m[0]+m[1])*.5,t=(cell.lo+cell.hi)*.5;
  SV nl=mul(add(cell.n[0],cell.n[1]),.5),nr=mul(add(cell.n[1],cell.n[2]),.5),nm=mul(add(nl,nr),.5);
  FBC right={t,cell.hi,{midpoint,m[1],l[2],cell.b[3]},{nm,nr,cell.n[2]},cell.depth+1};
  FBC left={cell.lo,t,{cell.b[0],l[0],m[0],midpoint},{cell.n[0],nl,nm},cell.depth+1};
  stack[size++]=right;stack[size++]=left;
 }
 return side;
}
__device__ int fh_finite_birth_counted(const float4* old,const float4* cur){
 unsigned kind,visited,boxes;int side=fh_finite_birth_side(old,cur,kind,visited,boxes);
 if(kind==2){atomicAdd(cloth_history_finite_counts,1ull);atomicAdd(cloth_history_finite_counts+(side?1:2),1ull);atomicAdd(cloth_history_finite_counts+3,(unsigned long long)visited);atomicMax(cloth_history_finite_counts+4,(unsigned long long)visited);}
 return side;
}
extern "C" __global__ void cloth_finite_birth_probe(const float4* old,const float4* cur,int* result,unsigned depth,unsigned intervals){
 if(blockIdx.x||threadIdx.x)return;unsigned kind,visited,boxes;
 result[0]=fh_finite_birth_side(old,cur,kind,visited,boxes,depth,intervals);result[1]=kind;result[2]=visited;result[3]=boxes;
}
