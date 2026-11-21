#include "native_contact_candidate_activation.h"
// Serial validation is intentional for this causal variant, not a performance claim.
// State: sticky failure code, accepted calls, visited rows, newly activated rows.
extern "C" __global__ void activate_generated_contact_candidates(
    const unsigned char* table, uint4* info, const unsigned char* constraints,
    const unsigned* count, const unsigned char* soft, unsigned nodes,
    const unsigned char* prep, const unsigned char* core, const unsigned char* arti,
    unsigned long long* state, unsigned* rejected) {
    if(blockIdx.x || threadIdx.x) return;
    if(state[0] || *rejected) {if(!state[0])state[0]=16; return;}
    unsigned code=count?ca_validate(table,info,constraints,*count,soft,nodes,prep,core,arti):17;
    if(code) {state[0]=code; atomicAdd(rejected,1u); return;}
    state[3]+=ca_activate(info,*count);state[2]+=*count;++state[1];
}
