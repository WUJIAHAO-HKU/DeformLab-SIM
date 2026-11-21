// Opt-in matched-binary TGS experiment. xyz residuals are in metres.
// Published float + residual is the logical position. Never derive velocity
// from the rounded world-position difference.
namespace pr {
__device__ float add(float old, double& residual, double increment) {
    double small=residual+increment;
    float next=float(double(old)+small);
    residual=small-(double(next)-double(old));
    return next;
}
__device__ void advance(float4& value,double3& residual,const double3 increment) {
    value.x=add(value.x,residual.x,increment.x);
    value.y=add(value.y,residual.y,increment.y);
    value.z=add(value.z,residual.z,increment.z);
}
}
// op=0: frame boundary, op=1: integration, op=2: contact.
extern "C" __global__ void residual_capture(char* sb,float4* cache,int op,unsigned* rejected) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x,n=field<unsigned>(sb,372);if(i>=n)return;
    cache[i]=field<float4*>(sb,96)[i];cache[n+i]=field<float4*>(sb,128)[i];
    if(!(cache[i].w>0)){atomicAdd(rejected,1u);return;}
    if(op==2){
        float4 x=field<float4*>(sb,272)[i];float scale;
        float denominator=fmaxf(x.w,1.f);
        // Match the pinned native contact update's reciprocal and products.
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(scale) : "f"(denominator));
        cache[2*n+i]=make_float4(__fmul_rn(x.x,scale),__fmul_rn(x.y,scale),__fmul_rn(x.z,scale),0);
    }
}
extern "C" __global__ void residual_publish(char* sb,float4* cache,double3* residual,int op,float dt,unsigned* rejected) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x,n=field<unsigned>(sb,372);if(i>=n)return;
    auto p=field<float4*>(sb,96);auto d=field<float4*>(sb,128);
    if(op==0){
        // Only TGS preIntegrate without position advancement is supported.
        float4 before=cache[i],after=p[i],delta=d[i];
        if(!(after.w>0) || before.x!=after.x || before.y!=after.y || before.z!=after.z || delta.x!=0 || delta.y!=0 || delta.z!=0)atomicAdd(rejected,1u);
        residual[n+i]=make_double3(0,0,0);return;
    }
    float4 a=cache[i],b=cache[n+i];a.w=p[i].w;b.w=d[i].w;
    float4 source=op==1?field<float4*>(sb,88)[i]:cache[2*n+i];
    double scale=op==1?double(dt):1.;
    double3 dx=make_double3(double(source.x)*scale,double(source.y)*scale,double(source.z)*scale);
    pr::advance(a,residual[i],dx);pr::advance(b,residual[n+i],dx);
    p[i]=a;d[i]=b;
}
extern "C" __global__ void merge_copy_velocity_residual(char* sb,float4* scratch,float invDt,double3* residual) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x,n=field<unsigned>(sb,372);if(i>=n)return;
    auto p=field<float4*>(sb,96);auto v=field<float4*>(sb,88);auto d=field<float4*>(sb,128);
    auto cv=field<float4*>(sb,176);auto counts=field<unsigned*>(sb,200);unsigned nt=field<unsigned>(sb,376);
    unsigned lo=i?counts[i-1]:0,hi=counts[i];float4 x=scratch[i],old=scratch[n+i],delta=scratch[2*n+i];
    double3 dv=make_double3(0,0,0);
    for(unsigned j=lo;j<hi;++j){float4 a=cv[4*nt+j];dv.x+=double(a.x)-old.x;dv.y+=double(a.y)-old.y;dv.z+=double(a.z)-old.z;}
    dv.x/=hi-lo;dv.y/=hi-lo;dv.z/=hi-lo;
    v[i]=make_float4(float(double(old.x)+dv.x),float(double(old.y)+dv.y),float(double(old.z)+dv.z),x.w);
    double3 dx=make_double3(dv.x/invDt,dv.y/invDt,dv.z/invDt);
    pr::advance(x,residual[i],dx);pr::advance(delta,residual[n+i],dx);p[i]=x;d[i]=delta;
}
