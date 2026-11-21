// Fixed axial coupon: simultaneous post-update position/velocity feasibility.
#include <cuda_runtime.h>
#include <math.h>
__device__ void correct_poststate_rows(double* rows,unsigned count,const double* prepared,
    const double* delta,unsigned nodes,double h,double rigid_v,unsigned terminal,unsigned* rejected) {
    if(!rows || !prepared || !delta || !nodes || nodes>4096 || count>1024 ||
       !isfinite(h) || h<=0. || !isfinite(rigid_v) || terminal>1){*rejected=1;return;}
    // Validate the entire publication first, including every arithmetic result.
    for(unsigned i=0;i<count;++i){
        const double* r=rows+6*i;if(r[0]==-1.)continue;
        if(!isfinite(r[0]) || r[0]<0. || r[0]>=nodes || r[0]!=double(unsigned(r[0])) ||
           (r[1]!=0. && r[1]!=1.)){*rejected=1;return;}
        for(unsigned j=2;j<6;++j)if(!isfinite(r[j])){*rejected=1;return;}
        unsigned node=unsigned(r[0]);double a=prepared[3*node+2],b=delta[3*node+2];
        double gap=r[4]-(r[1]==1.?1.:-1.)*(a+b);
        double G=gap/h+((r[1]==1. && !terminal)?rigid_v:0.);
        double joint=.5*G+.5*r[5];
        if(!isfinite(a) || !isfinite(b) || !isfinite(gap) || !isfinite(G) || !isfinite(joint)){*rejected=1;return;}
    }
    for(unsigned i=0;i<count;++i){
        double* r=rows+6*i;if(r[0]==-1.)continue;unsigned node=unsigned(r[0]);
        double gap=r[4]-(r[1]==1.?1.:-1.)*(prepared[3*node+2]+delta[3*node+2]);
        double G=gap/h+((r[1]==1. && !terminal)?rigid_v:0.);
        r[4]=gap;r[2]=fmin(G,.5*G+.5*r[5]);
    }
}

// A distinct entry prevents an old host from silently calling a changed ABI.
extern "C" __global__ void correct_contact_poststate_indirect(
    double* rows,const unsigned* count,const double* prepared,const double* residual,
    unsigned nodes,double h,unsigned long long frame,unsigned rounds,
    unsigned long long* state,const unsigned char* solver_core,unsigned terminal,unsigned* rejected) {
    if(blockIdx.x || threadIdx.x || !rejected || *rejected)return;
    if(!count || !state || !residual || !solver_core || !state[3] || state[0]!=frame ||
       state[2]!=nodes || rounds<2 || state[1]>=rounds || terminal>1 ||
       terminal!=unsigned(state[1]==rounds-1)){*rejected=1;return;}
    const float4* velocity=*reinterpret_cast<const float4* const*>(solver_core+40);
    if(!velocity){*rejected=1;return;}
    float4 v=velocity[3],w=velocity[7],dw=velocity[15];
    if(!isfinite(v.x) || !isfinite(v.y) || !isfinite(v.z) || w.x!=0.f || w.y!=0.f || w.z!=0.f ||
       dw.x!=0.f || dw.y!=0.f || dw.z!=0.f){*rejected=1;return;}
    correct_poststate_rows(rows,*count,prepared,residual+3*nodes,nodes,h,double(v.z),terminal,rejected);
    if(!*rejected)++state[1];
}
