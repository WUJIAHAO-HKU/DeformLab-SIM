// Read-only, fixed-capacity EE pre-solve trajectory and candidate observation.
static constexpr unsigned FHT_EPOCHS=131072,FHT_WORDS=48,FHT_TARGETS=2;
extern "C" {
 __device__ unsigned long long fht_keys[2];
 __device__ __align__(16) unsigned fht_records[FHT_EPOCHS*FHT_TARGETS*FHT_WORDS];
 __device__ unsigned long long fht_counts[3]; // records, overflow epochs, invalid target
}
__device__ unsigned fht_slots[2];
__device__ void fht_row(uint4 r,const uint4* triangles,unsigned nt,unsigned nodes,unsigned epoch){
 if(!epoch||epoch>FHT_EPOCHS)return;
 unsigned a=r.x&0xfffff,b=r.z&0xfffff,ea=r.w&16383,eb=(r.w>>14)&32767;
 if((r.x>>20)||(r.z>>20)||a>=nt||b>=nt||ea>=3||eb>=3)return;
 uint4 ta=triangles[a],tb=triangles[b];unsigned aa[3]={ta.x,ta.y,ta.z},bb[3]={tb.x,tb.y,tb.z};
 unsigned long long key=fh_key(make_uint4(aa[ea],aa[(ea+1)%3],bb[eb],bb[(eb+1)%3]),nodes);
 if(!key)return;
 for(unsigned k=0;k<2;++k)if(fht_keys[k]==key){
  unsigned* dst=fht_records+((epoch-1)*2+k)*FHT_WORDS;
  atomicOr(dst+6,r.w&((1u<<31)|(1u<<29)));atomicAdd(dst+7,1);
  if(r.w&(1u<<31))atomicAdd(dst+8,1);
  if((r.w&((1u<<31)|(1u<<29)))==((1u<<31)|(1u<<29)))atomicAdd(dst+9,1);
 }
}
__device__ void fht_slot(unsigned long long key,unsigned slot){
 for(unsigned k=0;k<2;++k)if(fht_keys[k]==key)atomicExch(fht_slots+k,slot+1);
}
__device__ void fht_capture(const float4* current,const float4* previous,const FH* history,unsigned capacity,unsigned epoch,unsigned nodes){
 if(!epoch||epoch>FHT_EPOCHS){atomicAdd(fht_counts+1,1ull);return;}
 for(unsigned k=0;k<2;++k){
  unsigned long long key=fht_keys[k];if(!key)continue;
  unsigned ids[4]={unsigned(key>>48),unsigned((key>>32)&65535),unsigned((key>>16)&65535),unsigned(key&65535)};
  bool valid=true;for(unsigned j=0;j<4;++j)valid&=ids[j]<nodes;
  if(!valid){atomicAdd(fht_counts+2,1ull);continue;}
  unsigned* dst=fht_records+((epoch-1)*2+k)*FHT_WORDS;
  dst[1]=epoch;dst[2]=unsigned(fh_diagnostic_frame);dst[3]=unsigned(fh_diagnostic_frame>>32);dst[4]=unsigned(key);dst[5]=unsigned(key>>32);
  unsigned slot=fht_slots[k];dst[10]=slot;
  if(slot){if(slot>capacity||history[slot-1].key!=key){atomicAdd(fht_counts+2,1ull);continue;}FH h=history[slot-1];dst[11]=h.state;dst[12]=unsigned(h.side);dst[13]=h.epoch;dst[14]=h.seen;}
  dst[15]=nodes;
  for(unsigned j=0;j<4;++j){reinterpret_cast<float4*>(dst+16)[j]=current[ids[j]];reinterpret_cast<float4*>(dst+32)[j]=previous[ids[j]];}
  __threadfence();dst[0]=2;atomicAdd(fht_counts,1ull);
 }
}
