// Owned GPU contact-row preparation for one volume above a finite static table.
// Does not write native buffers. A nonzero plan status forbids query/solve use.
// Boundary incident-map topology is validated once by the host reference.
#include <cuda_runtime.h>
#include <math.h>
static __device__ bool finite4(float4 p){return isfinite(p.x)&&isfinite(p.y)&&isfinite(p.z)&&isfinite(p.w);}
extern "C" __global__ void robot_surface_plan(
 const unsigned* count,const uint4* info,const unsigned* solver_ids,
 const float* constraints,const uint4* tets,const float4* position,const float4* delta,
 const uint4* boundary,unsigned boundary_count,unsigned nodes,unsigned ntets,
 double center_x,double center_y,double half_x,double half_y,double plane,double epsilon,
 unsigned capacity,unsigned* plan,uint4* selected){
 if(blockIdx.x||threadIdx.x)return;
 unsigned n=*count;plan[0]=0;plan[1]=n;plan[2]=0;plan[3]=~0u;
 if(n>=16384||nodes==0||nodes>65536||ntets==0||ntets>1048576||boundary_count>nodes||capacity<n||
    !isfinite(center_x)||!isfinite(center_y)||!isfinite(half_x)||!isfinite(half_y)||
    !isfinite(plane)||!isfinite(epsilon)||half_x<=0||half_y<=0||epsilon<0){plan[0]=1;return;}
 unsigned added=0,previous=~0u;
 for(unsigned i=0;i<boundary_count;++i){
  uint4 b=boundary[i];
  if(b.x>=nodes||b.y>=ntets||b.z>=4||(i&&b.x<=previous)){plan[0]=2;return;}previous=b.x;
  uint4 tet=tets[b.y];unsigned ids[4]={tet.x,tet.y,tet.z,tet.w};
  if(ids[b.z]!=b.x){plan[0]=2;return;}
  float4 x=position[b.x],d=delta[b.x];
  if(!finite4(x)||!finite4(d)){plan[0]=3;return;}
  if(!(fabs(double(x.x)-center_x)<half_x&&fabs(double(x.y)-center_y)<half_y&&double(x.z)<plane-epsilon))continue;
  if(d.w<=0||n+added>=capacity){plan[0]=4;return;}
  selected[added++]=b;
 }
 if(!added){plan[2]=n;return;}
 unsigned source=~0u;
 for(unsigned i=0;i<n;++i){
  uint4 in=info[i];if((in.z>>20)!=0){plan[0]=5;return;}
  if(in.x!=~0u||solver_ids[i]!=~0u)continue;
  unsigned base=(i/32)*800,k=i%32;
  const float* normal=constraints+base+384+4*k;
  if(normal[0]!=0||normal[1]!=0||normal[2]!=-1)continue;
  bool zero=true;for(unsigned f=0;f<3;++f)for(unsigned j=0;j<4;++j)zero=zero&&constraints[base+128*f+4*k+j]==0;
  if(!zero||constraints[base+512+4*k+3]!=0)continue;
  unsigned ti=in.z&0xfffff;if(ti>=ntets){plan[0]=6;return;}
  uint4 tet=tets[ti];unsigned ids[4]={tet.x,tet.y,tet.z,tet.w};double inferred=0;
  for(unsigned j=0;j<4;++j){
   if(ids[j]>=nodes){plan[0]=6;return;}
   float4 x=position[ids[j]],d=delta[ids[j]];float w=constraints[base+640+4*k+j];
   if(!finite4(x)||!finite4(d)||!isfinite(w)||!isfinite(normal[3])){plan[0]=3;return;}
   inferred+=double(w)*(double(x.z)-double(d.z));
  }
  inferred-=double(normal[3]);
  if(fabs(inferred-plane)<=2e-6&&source==~0u)source=i;
 }
 if(source==~0u){plan[0]=7;return;}
 plan[2]=n+added;plan[3]=source;
 for(unsigned j=0;j<added;++j)selected[j].w=source;
}
extern "C" __global__ void robot_surface_copy(
 const unsigned* plan,const uint4* info,const unsigned* sid,const float* constraints,
 const float* lambda,const float4* rigid,uint4* out_info,unsigned* out_sid,float* out_constraints,
 float* out_lambda,float4* out_rigid,unsigned* out_count){
 if(plan[0])return;
 unsigned n=plan[1],nc=plan[2],i=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
 if(i==0)*out_count=nc;
 for(unsigned j=i;j<nc;j+=stride){
  out_info[j]=j<n?info[j]:make_uint4(0,0,0,0);out_sid[j]=j<n?sid[j]:~0u;
  out_lambda[j]=j<n?lambda[j]:0;
  out_rigid[j]=j<n?rigid[j]:make_float4(0,0,0,0);
  out_rigid[nc+j]=j<n?rigid[n+j]:make_float4(0,0,0,0);
 }
 unsigned old_size=((n+31)/32)*800,new_size=((nc+31)/32)*800;
 for(unsigned j=i;j<new_size;j+=stride)out_constraints[j]=j<old_size?constraints[j]:0;
}
extern "C" __global__ void robot_surface_append(
 const unsigned* plan,const uint4* selected,const uint4* info,const float* constraints,
 const float4* position,const float4* delta,double plane,unsigned padding,
 uint4* out_info,float* out_constraints){
 if(plan[0])return;
 unsigned n=plan[1],nc=plan[2],j=blockIdx.x*blockDim.x+threadIdx.x;
 if(j>=nc-n)return;
 uint4 b=selected[j];unsigned dst=n+j,db=(dst/32)*800,dk=dst%32,sb=(b.w/32)*800,sk=b.w%32;
 for(unsigned f=0;f<6;++f)for(unsigned v=0;v<4;++v)out_constraints[db+128*f+4*dk+v]=constraints[sb+128*f+4*sk+v];
 out_constraints[db+768+dk]=constraints[sb+768+sk];
 double gap=double(position[b.x].z)-double(delta[b.x].z)-plane+(padding?1.:0.);
 out_constraints[db+384+4*dk+3]=float(gap);
 for(unsigned v=0;v<4;++v)out_constraints[db+640+4*dk+v]=(v==b.z?1.f:0.f);
 uint4 in=info[b.w];in.z=b.y;in.w&=0x7fffffff;out_info[dst]=in;
}
