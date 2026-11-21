#pragma once
// No learned parameters, success labels, or future trajectory values.
// Mode 0: D; 1: Periodic-4; 2: current gap; 3: gap+measured motion; 4: Dense.
struct EventDecision { bool refresh; unsigned reason; };
__host__ __device__ inline EventDecision event_policy(unsigned mode,unsigned round,unsigned age,double gap,double predicted,double budget){
 if(round==0)return {true,1u};
 if(mode==0||mode==4)return {true,2u};
 if(mode==1)return {(round%4)==0,(round%4)==0?2u:0u};
 unsigned reason=0;
 if(age>=4)reason|=16u;
 if(gap>budget)reason|=4u;
 if(mode==3&&predicted>budget)reason|=8u;
 return {reason!=0,reason};
}
