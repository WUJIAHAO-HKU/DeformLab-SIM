// Deterministic one-block construction. Each candidate keeps the serial
// arithmetic/order; block-local prefix packing preserves ascending node order.
static __device__ void parallel_surface_plan(SurfaceStreamState* s,unsigned frame,unsigned round){
 unsigned* plan=ptr<unsigned>(s->plan);unsigned* header=ptr<unsigned>(s->header);
 const unsigned n=*ptr<unsigned>(s->input[4]),tid=threadIdx.x;
 __shared__ unsigned flags[128],codes[128],errors[128],sources[128],added;
 if(tid==0){
  added=0;plan[0]=0;plan[1]=n;plan[2]=0;plan[3]=~0u;
  if(blockIdx.x||blockDim.x>128||header[0]!=0x53555246u||header[1]!=frame||header[2]!=round||header[3]!=1||header[4]!=s->nodes||header[5]!=s->boundary_count||header[6])plan[0]=900;
  else if(n>=16384||s->nodes==0||s->nodes>65536||s->tets==0||s->tets>1048576||s->boundary_count>s->nodes||s->capacity<n||!isfinite(s->cx)||!isfinite(s->cy)||!isfinite(s->hx)||!isfinite(s->hy)||!isfinite(s->plane)||!isfinite(s->epsilon)||s->hx<=0||s->hy<=0||s->epsilon<0)plan[0]=1;
 }
 __syncthreads();if(plan[0])return;
 const uint4* boundary=ptr<uint4>(s->boundary);const uint4* tets=ptr<uint4>(s->input[5]);const float4* position=ptr<float4>(s->input[6]);const float4* delta=ptr<float4>(s->input[7]);
 const unsigned* seen=ptr<unsigned>(s->seen);const float* saved=ptr<float>(s->cache);uint4* selected=ptr<uint4>(s->selected);
 for(unsigned base=0;base<s->boundary_count;base+=blockDim.x){
  unsigned i=base+tid;uint4 b=make_uint4(0,0,0,0);unsigned code=0,choose=0;
  if(i<s->boundary_count){
   b=boundary[i];
   if(b.x>=s->nodes||b.y>=s->tets||b.z>=4||(i&&b.x<=boundary[i-1].x))code=2;
   else {
    uint4 tet=tets[b.y];unsigned ids[4]={tet.x,tet.y,tet.z,tet.w};
    if(ids[b.z]!=b.x)code=2;
    else {
     float4 x=position[b.x],d=delta[b.x];
     if(!finite4(x)||!finite4(d))code=3;
     else {
      bool inside=fabs(double(x.x)-s->cx)<s->hx&&fabs(double(x.y)-s->cy)<s->hy;
      if(seen[b.x]>1||!isfinite(saved[b.x])||saved[b.x]<0||(!seen[b.x]&&saved[b.x]!=0))code=901;
      else if(seen[b.x]&&!inside)code=902;
      else if(seen[b.x]||(inside&&double(x.z)<s->plane-s->epsilon)){choose=1;if(d.w<=0)code=4;}
     }
    }
   }
  }
  flags[tid]=choose;codes[tid]=code;__syncthreads();
  if(tid==0){
   for(unsigned lane=0;lane<blockDim.x&&base+lane<s->boundary_count;++lane){
    if(codes[lane]){plan[0]=codes[lane];break;}
    if(flags[lane]){if(n+added>=s->capacity){plan[0]=4;break;}flags[lane]=added++;}
    else flags[lane]=~0u;
   }
  }
  __syncthreads();if(plan[0])return;
  if(i<s->boundary_count&&flags[tid]!=~0u)selected[flags[tid]]=b;
  __syncthreads();
 }
 if(!added){if(tid==0){plan[2]=n;header[3]=2;}return;}
 const uint4* info=ptr<uint4>(s->input[0]);const unsigned* sid=ptr<unsigned>(s->input[1]);const float* constraints=ptr<float>(s->input[2]);
 unsigned bad=~0u,code=0,source=~0u;
 for(unsigned i=tid;i<n;i+=blockDim.x){
  uint4 in=info[i];unsigned err=0;
  if((in.z>>20)!=0)err=5;
  else if(in.x==~0u&&sid[i]==~0u){
   unsigned base=(i/32)*800,k=i%32;const float* normal=constraints+base+384+4*k;
   if(normal[0]==0&&normal[1]==0&&normal[2]==-1){
    bool zero=true;for(unsigned f=0;f<3;++f)for(unsigned j=0;j<4;++j)zero=zero&&constraints[base+128*f+4*k+j]==0;
    if(zero&&constraints[base+512+4*k+3]==0){
     unsigned ti=in.z&0xfffff;
     if(ti>=s->tets)err=6;
     else {
      uint4 tet=tets[ti];unsigned ids[4]={tet.x,tet.y,tet.z,tet.w};double inferred=0;
      for(unsigned j=0;j<4;++j){
       if(ids[j]>=s->nodes){err=6;break;}
       float4 x=position[ids[j]],d=delta[ids[j]];float w=constraints[base+640+4*k+j];
       if(!finite4(x)||!finite4(d)||!isfinite(w)||!isfinite(normal[3])){err=3;break;}
       inferred+=double(w)*(double(x.z)-double(d.z));
      }
      inferred-=double(normal[3]);
      if(!err&&fabs(inferred-s->plane)<=2e-6&&source==~0u)source=i;
     }
    }
   }
  }
  if(err){bad=i;code=err;break;}
 }
 errors[tid]=bad;codes[tid]=code;sources[tid]=source;__syncthreads();
 if(tid==0){
  unsigned first=~0u,chosen=~0u,err=0;
  for(unsigned lane=0;lane<blockDim.x;++lane){if(errors[lane]<first){first=errors[lane];err=codes[lane];}if(sources[lane]<chosen)chosen=sources[lane];}
  if(first!=~0u)plan[0]=err;
  else if(chosen==~0u)plan[0]=7;
  else {plan[2]=n+added;plan[3]=chosen;header[3]=2;}
 }
 __syncthreads();if(plan[0])return;
 for(unsigned j=tid;j<added;j+=blockDim.x)selected[j].w=plan[3];
}
