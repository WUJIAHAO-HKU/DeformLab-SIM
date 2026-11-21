// Appended to the hash-pinned gap arithmetic after converting its entry to a
// device helper. Fixed axial coupon only; this is not a general contact model.
template<class T> __device__ T gap_field(char* p,unsigned offset) {
    return *reinterpret_cast<T*>(p+offset);
}
extern "C" __global__ void capture_contact_gap_origin(
    char* sb,const double* residual,double* prepared,unsigned nodes,
    unsigned long long frame,unsigned rounds,unsigned long long* state,unsigned* rejected) {
    if(blockIdx.x || threadIdx.x || !rejected || *rejected)return;
    if(!sb || !residual || !prepared || !state || !nodes || nodes>4096 ||
       !frame || !rounds || gap_field<unsigned>(sb,372)!=nodes) {*rejected=1;return;}
    if(state[3] && (state[0]+1!=frame || state[1]!=rounds || state[2]!=nodes)) {*rejected=1;return;}
    const float4* delta=gap_field<float4*>(sb,128);
    if(!delta){*rejected=1;return;}
    // Whole-frame validation before publishing either cache or stamp.
    for(unsigned i=0;i<nodes;++i){
        if(delta[i].x!=0.f || delta[i].y!=0.f || delta[i].z!=0.f){*rejected=1;return;}
        for(unsigned j=0;j<3;++j)
            if(!isfinite(residual[3*i+j]) || residual[3*(nodes+i)+j]!=0.) {*rejected=1;return;}
    }
    for(unsigned i=0;i<3*nodes;++i)prepared[i]=residual[i];
    state[0]=frame;state[1]=0;state[2]=nodes;state[3]=1;
}
extern "C" __global__ void correct_contact_gap_indirect(
    double* rows,const unsigned* count,const double* prepared,const double* residual,
    unsigned nodes,double h,unsigned long long frame,unsigned rounds,
    unsigned long long* state,unsigned* rejected) {
    if(blockIdx.x || threadIdx.x || !rejected || *rejected)return;
    if(!count || !state || !residual || !state[3] || state[0]!=frame ||
       state[2]!=nodes || !rounds || state[1]>=rounds){*rejected=1;return;}
    correct_contact_gap_value(rows,*count,prepared,residual+3*nodes,nodes,h,1,rejected);
    if(!*rejected)++state[1];
}
