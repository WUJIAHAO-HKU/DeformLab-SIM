// Owned probe only: certify the supplied polyline, never an unobserved kernel path.
#include "cloth_finite_history_birth_v2.cu"
extern "C" __global__ void cloth_polyline_birth_probe(const float4* path,unsigned points,unsigned max_segments,unsigned max_work,int* result){
 if(blockIdx.x||threadIdx.x)return;
 result[0]=0;result[1]=0;result[2]=0;result[3]=1;
 if(!path||points<2||points>129||!max_segments||max_segments>128||points-1>max_segments||!max_work||max_work>256)return;
 unsigned used=0;int side=0;
 for(unsigned i=0;i<points-1;++i){
  if(used>=max_work){result[3]=2;return;}
  float4 old[4],cur[4];for(unsigned j=0;j<4;++j){old[j]=path[i*4+j];cur[j]=path[(i+1)*4+j];}
  unsigned kind,visited,boxes,remaining=max_work-used;unsigned bound=remaining<64?remaining:64;
  side=fh_finite_birth_side(old,cur,kind,visited,boxes,12,bound);used+=visited?visited:1;
  result[1]=int(i+1);result[2]=int(used);
  if(!side||used>max_work){result[3]=3;return;}
 }
 result[0]=side;result[3]=0;
}
