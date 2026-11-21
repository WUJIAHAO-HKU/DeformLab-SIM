// Unpublished owned-buffer primitive. Runtime integration must supply the
// residual captured at contact preparation and the current frame-delta residual.
#include <cuda_runtime.h>
#include <math.h>
extern "C" __global__ void correct_contact_gap_residual(
    double* rows,unsigned count,const double* prepared_world_residual,
    const double* current_delta_residual,unsigned nodes,double h,unsigned mode,unsigned* rejected) {
    if(blockIdx.x || threadIdx.x || !rejected)return;
    if(*rejected)return;
    if(mode==0)return;
    if(mode!=1 || !rows || !prepared_world_residual || !current_delta_residual ||
       count>1024 || !nodes || nodes>4096 || !isfinite(h) || h<=0.) {*rejected=1;return;}
    // Validate the complete table before changing any row, including finite
    // computed gaps. No partially published correction on invalid input.
    for(unsigned i=0;i<count;++i){
        const double* r=rows+6*i;
        if(r[0]==-1.)continue;
        if(!isfinite(r[0]) || r[0]<0. || r[0]>=nodes || r[0]!=double(unsigned(r[0])) ||
           (r[1]!=0. && r[1]!=1.)) {*rejected=1;return;}
        for(unsigned j=2;j<6;++j)if(!isfinite(r[j])){*rejected=1;return;}
        unsigned node=unsigned(r[0]);double a=prepared_world_residual[3*node+2],b=current_delta_residual[3*node+2];
        double gap=r[4]-(r[1]==1.?1.:-1.)*(a+b);
        if(!isfinite(a) || !isfinite(b) || !isfinite(gap) || !isfinite(gap/h)){*rejected=1;return;}
    }
    for(unsigned i=0;i<count;++i){
        double* r=rows+6*i;if(r[0]==-1.)continue;unsigned node=unsigned(r[0]);
        double gap=r[4]-(r[1]==1.?1.:-1.)*(prepared_world_residual[3*node+2]+current_delta_residual[3*node+2]);
        r[4]=gap;r[2]=fmin(gap/h,r[5]);
    }
}
