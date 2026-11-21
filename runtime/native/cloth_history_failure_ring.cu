// Diagnostic-only bounded pre-EE snapshots. FH is the 24-byte production cache entry.
// Inputs and solver arrays are read-only. Caller allocates slots*nodes float4s for
// each position array and slots*capacity FHs, then launches publish on the same
// ordered stream after capture. These kernels do not imply native integration.
static_assert(sizeof(FH)==24,"History ABI mismatch");
__device__ unsigned fhr_limit(const unsigned* rejection,const unsigned* decode){
 unsigned a=rejection[0],b=decode[0];
 if((a!=0&&a!=2)||(b!=0&&b!=2))return ~0u;
 unsigned ea=a?rejection[4]:0,eb=b?decode[4]:0;
 if((a&&!ea)||(b&&!eb))return ~0u;
 return ea&&eb?(ea<eb?ea:eb):(ea?ea:eb);
}
__device__ bool fhr_valid(unsigned nodes,unsigned capacity,unsigned slots,unsigned epoch){
 return nodes&&nodes<=4096&&capacity&&capacity<=32768&&slots&&slots<=256&&epoch;
}
extern "C" __global__ void cloth_history_ring_capture(
 const float4* current,const float4* previous,const FH* history,
 unsigned nodes,unsigned capacity,unsigned slots,unsigned epoch,unsigned long long frame,
 const unsigned* rejection,const unsigned* decode,unsigned* metadata,
 float4* positions,float4* roots,FH* entries,unsigned long long* counts){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x,stride=gridDim.x*blockDim.x;
 unsigned limit=fhr_limit(rejection,decode);
 if(!fhr_valid(nodes,capacity,slots,epoch)||limit==~0u){if(!i)atomicAdd(counts+3,1ull);return;}
 if(limit&&epoch>limit){if(!i)atomicAdd(counts+2,1ull);return;}
 unsigned slot=(epoch-1)%slots;unsigned* m=metadata+slot*8;
 if(!i){m[0]=0;m[1]=epoch;m[2]=unsigned(frame);m[3]=unsigned(frame>>32);m[4]=nodes;m[5]=capacity;m[6]=1;m[7]=0;atomicAdd(counts,1ull);}
 for(unsigned j=i;j<nodes;j+=stride){positions[size_t(slot)*nodes+j]=current[j];roots[size_t(slot)*nodes+j]=previous[j];}
 for(unsigned j=i;j<capacity;j+=stride)entries[size_t(slot)*capacity+j]=history[j];
}
extern "C" __global__ void cloth_history_ring_publish(
 unsigned nodes,unsigned capacity,unsigned slots,unsigned epoch,
 const unsigned* rejection,const unsigned* decode,unsigned* metadata,unsigned long long* counts){
 if(blockIdx.x||threadIdx.x)return;
 unsigned limit=fhr_limit(rejection,decode);
 if(!fhr_valid(nodes,capacity,slots,epoch)||limit==~0u){atomicAdd(counts+3,1ull);return;}
 if(limit&&epoch>limit)return;
 unsigned* m=metadata+((epoch-1)%slots)*8;
 if(m[0]||m[1]!=epoch||m[4]!=nodes||m[5]!=capacity||m[6]!=1){atomicAdd(counts+3,1ull);return;}
 __threadfence();m[0]=2;atomicAdd(counts+1,1ull);
}
