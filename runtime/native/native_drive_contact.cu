// Experimental contact/drive Schur stages for the pinned Cartesian coupon.
// The existing native contact decoder, solver, publication and transport remain
// separate. These stages supply the effective response and the paired drive J.
#include "native_rank_one_adapter.cu"

// state: inverse response, beta, contact J, requested drive J, old drive J,
// representable drive J, constitutive residual, velocity-pass marker.
extern "C" __global__ void prepare_drive_contact(
    const unsigned char* arti,float h,unsigned velocity_pass,double* inputs,
    double* state,float* drive_delta,unsigned* bad) {
    if(blockIdx.x || threadIdx.x)return;
    for(unsigned i=0;i<8;++i)state[i]=0.;
    for(unsigned i=0;i<3;++i)drive_delta[i]=0.f;
    if(*bad)return;
    if(!arti || !isfinite(h) || h<=0.f || velocity_pass>1 ||
       native_field<unsigned>(arti,44)!=1 || native_field<unsigned>(arti,80)!=1 ||
       native_field<unsigned>(arti,248)!=4 || native_field<unsigned>(arti,252)!=3) {reject(bad);return;}
    auto blocks=native_field<const unsigned char*>(arti,128);
    auto links=native_field<const unsigned char*>(arti,136);
    auto dofs=native_field<const unsigned char*>(arti,184);
    if(!blocks || !links || !dofs || native_field<unsigned char>(blocks,7040)!=0 ||
       native_field<unsigned>(blocks,3840) || native_field<unsigned>(blocks,3968) ||
       native_field<unsigned>(blocks,4096) || native_field<unsigned>(blocks,7072)) {reject(bad);return;}
    for(unsigned j=0;j<4;++j) {
        auto l=links+j*29184;
        if(native_field<unsigned>(l,22016) || native_field<unsigned>(l,22272) ||
           (j && (native_field<unsigned>(l,28672)!=j-1 || native_field<unsigned char>(l,28800)!=1 ||
                  native_field<unsigned>(l,28544)!=j-1))) {reject(bad);return;}
    }
    auto z=dofs+2*12160;
    double stiffness=native_field<float>(z,10816),damping=native_field<float>(z,10944);
    double response=native_field<float>(z,7872),old=native_field<float>(z,8896);
    if(!isfinite(stiffness) || stiffness<0. || !isfinite(damping) || damping<0. ||
       !isfinite(old) || !isfinite(response) || fabs(response-5.)>2e-6 ||
       native_field<unsigned>(z,9280)!=0 || inputs[IN+1]!=5.) {reject(bad);return;}
    // Retain the already-validated physical inverse response 5 for this fixture.
    // Terminal velocity iteration has no positional drive increment.
    double a=double(h)*(damping+double(h)*stiffness);
    double beta=velocity_pass?0.:a/(.2+a);
    double effective=velocity_pass?5.:1./(.2+a);
    if(!isfinite(beta) || !isfinite(effective) || effective<=0.) {reject(bad);return;}
    inputs[IN+1]=effective;state[0]=effective;state[1]=beta;
    state[4]=old;state[7]=velocity_pass;
}

__device__ float drive_ftz(float x) {
    unsigned u=__float_as_uint(x);
    return (u&0x7fffffffu)<0x00800000u?__uint_as_float(u&0x80000000u):x;
}
extern "C" __global__ void reduce_contact_drive_increment(
    const double* rows,const unsigned* count,const float* target,
    double* state,float* drive_delta,unsigned* bad) {
    if(blockIdx.x || threadIdx.x)return;
    if(*bad)return;
    if(*count>ROW_CAP){reject(bad);return;}
    double impulse=0.;
    for(unsigned i=0;i<*count;++i) {
        if(rows[6*i]<0. || rows[6*i+1]!=1.)continue;
        float previous=float(rows[6*i+3]),next=target[i];
        if(!isfinite(previous) || !isfinite(next) || previous<0.f || next<0.f) {reject(bad);return;}
        // Match the native contact kernel's actual float32/FTZ subtraction,
        // rather than summing ideal double-precision target multipliers.
        impulse+=double(drive_ftz(__fsub_rn(drive_ftz(next),drive_ftz(previous))));
    }
    double wanted=-state[1]*impulse;
    float request=float(wanted),old=float(state[4]);
    float updated=drive_ftz(__fadd_rn(drive_ftz(old),drive_ftz(request)));
    float effective=drive_ftz(__fsub_rn(updated,drive_ftz(old)));
    if(!isfinite(impulse) || !isfinite(wanted) || !isfinite(request) || !isfinite(updated) || !isfinite(effective)) {reject(bad);return;}
    drive_delta[0]=drive_delta[1]=0.f;drive_delta[2]=request;
    state[2]=impulse;state[3]=wanted;state[5]=double(effective);
    state[6]=double(effective)-wanted;
}
