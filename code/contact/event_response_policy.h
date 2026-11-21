#pragma once
// 0 D, 1 Periodic-4, 2 gap, 3 gap+motion, 4 Dense,
// 5 material response, 6 material response + final-round membership refresh.
struct EventDecision { bool refresh; unsigned reason; };
__host__ __device__ inline EventDecision response_policy(unsigned mode,unsigned round,unsigned age,double gap,double predicted,double response,double budget,unsigned final_round){
 if(round==0)return {true,1u};
 if(mode==0||mode==4)return {true,2u};
 if(mode==1)return {(round%4)==0,(round%4)==0?2u:0u};
 unsigned reason=0;
 if(age>=4)reason|=16u;
 if(gap>budget)reason|=4u;
 if(mode>=3&&predicted>budget)reason|=8u;
 if(mode>=5&&response>budget)reason|=64u;
 if(mode==7&&round==23)reason|=32u;
 return {reason!=0,reason};
}
