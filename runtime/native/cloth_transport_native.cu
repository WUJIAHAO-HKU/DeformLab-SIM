// Decode pinned single-cloth native EE rows into owned geometry-key history.
#include "cloth_ee_feature_history_transport.cu"
extern "C" __global__ void cloth_ee_history_resolve_native(const uint4* rows,unsigned count,const uint4* triangles,unsigned nt,unsigned nodes,FH* history,unsigned capacity,unsigned epoch,unsigned* slots,unsigned long long* counters){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 slots[i]=~0u;atomicAdd(counters,1ull);
 if(!capacity||(capacity&(capacity-1))||!epoch||nodes>65536){atomicAdd(counters+4,1ull);return;}
 uint4 r=rows[i];if(!(r.w&(1u<<31))){atomicAdd(counters+1,1ull);return;}
 unsigned first=r.x&0xfffff,second=r.z&0xfffff,ea=r.w&16383,eb=(r.w>>14)&32767;
 if((r.x>>20)||(r.z>>20)||first>=nt||second>=nt||ea>2||eb>2){atomicAdd(counters+4,1ull);return;}
 uint4 a=triangles[first],b=triangles[second];unsigned av[3]={a.x,a.y,a.z},bv[3]={b.x,b.y,b.z};uint4 ids=make_uint4(av[ea],av[(ea+1)%3],bv[eb],bv[(eb+1)%3]);
 if(ids.x>=nodes||ids.y>=nodes||ids.z>=nodes||ids.w>=nodes||ids.x==ids.y||ids.z==ids.w){atomicAdd(counters+4,1ull);return;}
 if(ids.x==ids.z||ids.x==ids.w||ids.y==ids.z||ids.y==ids.w){atomicAdd(counters+2,1ull);return;}
 unsigned long long key=fh_key(ids,nodes);unsigned home=fh_hash(key)&(capacity-1);
 for(unsigned j=0;j<capacity;++j){unsigned at=(home+j)&(capacity-1);unsigned long long old=atomicCAS(&history[at].key,0ull,key);
  if(old==0||old==key){slots[i]=at;atomicMax(&history[at].seen,epoch);atomicAdd(counters+3,1ull);return;}
 }
 atomicAdd(counters+4,1ull);
}
