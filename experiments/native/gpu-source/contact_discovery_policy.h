// Admission policy only. Contact arithmetic and response history are unchanged.
#ifndef COMPONENT_DISCOVERY_PERIOD
#error Explicit discovery period required
#endif
#ifndef COMPONENT_UNIFORM_DENOMINATOR
#error Explicit uniform coverage denominator required
#endif
static_assert(COMPONENT_DISCOVERY_PERIOD==1 || COMPONENT_DISCOVERY_PERIOD==2 || COMPONENT_DISCOVERY_PERIOD==4,"discovery period");
static_assert(COMPONENT_UNIFORM_DENOMINATOR==1 || COMPONENT_UNIFORM_DENOMINATOR==2 || COMPONENT_UNIFORM_DENOMINATOR==4,"uniform denominator");
static __device__ __forceinline__ bool contact_discovery_refresh(unsigned round){
 return (round % COMPONENT_DISCOVERY_PERIOD)==0;
}
static __device__ __forceinline__ bool contact_uniform_member(unsigned node){
#if COMPONENT_UNIFORM_DENOMINATOR==1
 return true;
#else
 // Fixed topology-independent hash; no task outcome or geometry feedback.
 node^=node>>16;node*=0x7feb352du;node^=node>>15;node*=0x846ca68bu;node^=node>>16;
 return (node & (COMPONENT_UNIFORM_DENOMINATOR-1))==0;
#endif
}
