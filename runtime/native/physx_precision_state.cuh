// Experimental zero-gravity TGS state transport. No global momentum projection.
extern "C" __global__ void cache_partition_precision(char* sb,double4* cache,unsigned part,double3* residual,float4* saved,double4* precise_p,double4* precise_v){
    auto prefix=field<unsigned*>(sb,208);unsigned t=(part?prefix[part-1]:0)+blockIdx.x*blockDim.x+threadIdx.x;if(t>=prefix[part])return;
    unsigned nt=field<unsigned>(sb,376),n=field<unsigned>(sb,372);uint4 ids=field<uint4*>(sb,400)[t];ids.x&=0x7fffffff;
    auto p=field<float4*>(sb,96);auto v=field<float4*>(sb,88);auto counts=field<unsigned*>(sb,200);
    for(int k=0;k<4;++k){unsigned id=(&ids.x)[k];
        if(id<0x7fffffff){float4 a=p[id],b=v[id];double3 rp=residual[id],rv=residual[2*n+id];
            double w=double(saved[id].w)*(counts[id]-(id?counts[id-1]:0));
            cache[8*t+k]=make_double4(double(a.x)+rp.x,double(a.y)+rp.y,double(a.z)+rp.z,w);
            cache[8*t+4+k]=make_double4(double(b.x)+rv.x,double(b.y)+rv.y,double(b.z)+rv.z,w);
        }else{cache[8*t+k]=precise_p[t+k*nt];cache[8*t+4+k]=precise_v[t+k*nt];}
    }
}
extern "C" __global__ void precision_capture(char* sb,float4* cache,int op,unsigned* rejected){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x,n=field<unsigned>(sb,372);if(i>=n)return;
    cache[i]=field<float4*>(sb,96)[i];cache[n+i]=field<float4*>(sb,128)[i];cache[3*n+i]=field<float4*>(sb,88)[i];
    if(!(cache[i].w>0)){atomicAdd(rejected,1u);return;}
    if(op==2){float4 x=field<float4*>(sb,272)[i];float scale,den=fmaxf(x.w,1.f);
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(scale) : "f"(den));
        float a,b,c;asm("mul.ftz.f32 %0, %1, %2;":"=f"(a):"f"(x.x),"f"(scale));
        asm("mul.ftz.f32 %0, %1, %2;":"=f"(b):"f"(x.y),"f"(scale));asm("mul.ftz.f32 %0, %1, %2;":"=f"(c):"f"(x.z),"f"(scale));
        cache[2*n+i]=make_float4(a,b,c,0);
    }
}
extern "C" __global__ void precision_publish(char* sb,float4* cache,double3* residual,int op,float parameter,unsigned* rejected){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x,n=field<unsigned>(sb,372);if(i>=n)return;
    auto p=field<float4*>(sb,96);auto d=field<float4*>(sb,128);auto v=field<float4*>(sb,88);
    float4 oldv=cache[3*n+i];
    if(op!=2 && (oldv.x!=v[i].x || oldv.y!=v[i].y || oldv.z!=v[i].z)){atomicAdd(rejected,1u);return;}
    if(op==0){float4 a=cache[i],b=p[i],delta=d[i];
        if(!(b.w>0) || a.x!=b.x || a.y!=b.y || a.z!=b.z || delta.x!=0 || delta.y!=0 || delta.z!=0)atomicAdd(rejected,1u);
        residual[n+i]=make_double3(0,0,0);return;}
    float4 a=cache[i],b=cache[n+i];a.w=p[i].w;b.w=d[i].w;double3 dx;
    if(op==1){double3 rv=residual[2*n+i];dx=make_double3((double(v[i].x)+rv.x)*parameter,(double(v[i].y)+rv.y)*parameter,(double(v[i].z)+rv.z)*parameter);}
    else{float4 x=cache[2*n+i];dx=make_double3(x.x,x.y,x.z);
        double3 dv=make_double3(dx.x*parameter,dx.y*parameter,dx.z*parameter);
        oldv.w=v[i].w;pr::advance(oldv,residual[2*n+i],dv);v[i]=oldv;}
    pr::advance(a,residual[i],dx);pr::advance(b,residual[n+i],dx);p[i]=a;d[i]=b;
}
extern "C" __global__ void merge_copy_precision(char* sb,float4* scratch,float invdt,double3* residual,double4* precise_v){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x,n=field<unsigned>(sb,372);if(i>=n)return;
    auto p=field<float4*>(sb,96);auto v=field<float4*>(sb,88);auto d=field<float4*>(sb,128);auto counts=field<unsigned*>(sb,200);unsigned nt=field<unsigned>(sb,376);
    float4 x=scratch[i],old=scratch[n+i],delta=scratch[2*n+i];double3 rv=residual[2*n+i];
    double3 base=make_double3(double(old.x)+rv.x,double(old.y)+rv.y,double(old.z)+rv.z),dv=make_double3(0,0,0);
    unsigned lo=i?counts[i-1]:0,hi=counts[i];
    for(unsigned j=lo;j<hi;++j){double4 a=precise_v[4*nt+j];dv.x+=a.x-base.x;dv.y+=a.y-base.y;dv.z+=a.z-base.z;}
    dv.x/=hi-lo;dv.y/=hi-lo;dv.z/=hi-lo;
    old.w=x.w;pr::advance(old,residual[2*n+i],dv);v[i]=old;
    double3 dx=make_double3(dv.x/invdt,dv.y/invdt,dv.z/invdt);
    pr::advance(x,residual[i],dx);pr::advance(delta,residual[n+i],dx);p[i]=x;d[i]=delta;
}
