// Version-pinned, world-z, frictionless one-volume Cartesian coupon adapter.
// All grouping and state assembly occur on device. Never dereference host snapshots.
#include "terminal_rank_one_contact.cu"
constexpr unsigned ROW_CAP = 1024;
template<class T> __device__ T native_field(const unsigned char* p, unsigned offset) {
    return *reinterpret_cast<const T*>(p + offset);
}
__device__ void reject(unsigned* bad) { atomicExch(bad, 1u); }
// Row scratch: physical node (-1 for inactive), face, c, old, gap, relative v.
extern "C" __global__ void decode_rank_one_rows(
    const unsigned char* table, const uint4* info, const unsigned char* constraints,
    const unsigned* count, const unsigned char* prep, const unsigned char* core,
    const unsigned char* arti, const float* lambda, float h,
    const unsigned char* soft, unsigned nodes, double* rows, unsigned* bad) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x, n=*count;
    if (n>ROW_CAP || !isfinite(h) || h<=0 || !nodes || nodes>4096 ||
        native_field<unsigned>(soft,372)!=nodes || native_field<unsigned>(arti,44)!=1 ||
        native_field<unsigned>(arti,248)!=4) { reject(bad); return; }
    if (i>=n) return;
    double* out=rows+6*i;out[0]=-1.;out[1]=0.;out[2]=0.;out[3]=lambda[i];out[4]=0.;out[5]=0.;
    if (!isfinite(lambda[i]) || lambda[i]<0) { reject(bad); return; }
    uint4 row=info[i];
    if (!(row.w&0x80000000u)) return;
    unsigned tet=row.z&0xfffffu, body=row.z>>20;
    if (table+size_t(body)*432!=soft || tet>=native_field<unsigned>(soft,376)) { reject(bad); return; }
    bool top=(row.y&1u)!=0;
    if ((top && (row.x>=64u || row.y!=7u)) || (!top && row.x!=0xffffffffu)) { reject(bad); return; }
    const unsigned char* block=constraints+(i/32)*3200+(i%32)*16;
    float4 normal=native_field<float4>(block,1536), bc=native_field<float4>(block,2560);
    if (normal.x!=0 || normal.y!=0 || normal.z!=(top?1.f:-1.f) || !isfinite(normal.w)) { reject(bad); return; }
    float weights[4]={bc.x,bc.y,bc.z,bc.w};int selected=-1;
    for(int j=0;j<4;++j) {
        if(weights[j]==1.f && selected==-1) selected=j;
        else if(weights[j]!=0.f) { reject(bad);return; }
    }
    if(selected<0) { reject(bad);return; }
    auto tets=native_field<const uint4*>(soft,80);uint4 ids=tets[tet];unsigned idx[4]={ids.x,ids.y,ids.z,ids.w};
    unsigned node=idx[selected];if(node>=nodes) { reject(bad);return; }
    auto pos=native_field<const float4*>(soft,96);auto v=native_field<const float4*>(soft,88);
    auto dp=native_field<const float4*>(soft,128);auto raw=native_field<const float4*>(soft,272);
    if(!isfinite(pos[node].w) || pos[node].w<=0 || raw[node].w<=0 || !isfinite(raw[node].w)) { reject(bad);return; }
    double rv=0.,rd=0.;
    if(top) {
        auto indices=native_field<const unsigned*>(prep,272);
        if(indices[row.x]!=0) { reject(bad);return; }
        auto velocity=native_field<const float4*>(core,40);
        float4 angular=velocity[4+3], angular_delta=velocity[12+3];
        if(angular.x!=0 || angular.y!=0 || angular.z!=0 || angular_delta.x!=0 || angular_delta.y!=0 || angular_delta.z!=0) { reject(bad);return; }
        rv=velocity[3].z;rd=velocity[8+3].z;
    }
    double gap=double(normal.w)+(rd-double(dp[node].z))*double(normal.z);
    double relative=(rv-double(v[node].z))*double(normal.z);
    double c=fmin(gap/double(h),relative);
    if(!isfinite(gap) || !isfinite(relative) || !isfinite(c)) { reject(bad);return; }
    out[0]=node;out[1]=top?1.:0.;out[2]=c;out[4]=gap;out[5]=relative;
}

// Per-node sums retain native row order; no floating atomic accumulation.
extern "C" __global__ void group_rank_one_nodes(const double* rows, const unsigned* count,
    const float4* positions, unsigned nodes, double* groups, unsigned* bad) {
    unsigned node=blockIdx.x*blockDim.x+threadIdx.x;
    if(*bad || node>=nodes) return;
    for(unsigned side=0;side<2;++side) {
        double c=0.,old=0.;unsigned found=0;
        for(unsigned i=0;i<*count;++i) if(rows[6*i]==double(node) && rows[6*i+1]==double(side)) {
            c=found?fmin(c,rows[6*i+2]):rows[6*i+2];old+=rows[6*i+3];++found;
        }
        double* g=groups+4*(side*nodes+node);
        g[0]=positions[node].w;g[1]=c;g[2]=old;g[3]=found;
        if(!isfinite(old)) reject(bad);
    }
}

// Ascending physical-node IDs match the independent reference, every launch.
extern "C" __global__ void compact_rank_one_groups(const double* groups, unsigned nodes,
    float h, double* inputs, int* node_to_group, unsigned* bad) {
    if(blockIdx.x || threadIdx.x) return;
    if(!nodes || nodes>4096) {reject(bad);inputs[0]=-1.;inputs[IN]=-1.;return;}
    for(unsigned side=0;side<2;++side) {
        double* a=inputs+side*IN;
        for(unsigned k=0;k<IN;++k) a[k]=0.;
        a[1]=side?5.:0.;a[2]=h;
        unsigned n=0;
        for(unsigned node=0;node<nodes;++node) {
            node_to_group[side*nodes+node]=-1;
            if(*bad) continue;
            const double* g=groups+4*(side*nodes+node);
            if(g[3]==0) continue;
            if(groups[4*((1-side)*nodes+node)+3]!=0 || n>=CAP) {reject(bad);continue;}
            node_to_group[side*nodes+node]=n;
            a[3+4*n]=g[0];a[4+4*n]=g[1];a[5+4*n]=g[2];++n;
        }
        a[0]=n;
    }
    if(*bad) {inputs[0]=-1.;inputs[IN]=-1.;}
}

extern "C" __global__ void publish_rank_one_targets(const double* rows,const unsigned* count,
    unsigned nodes,const double* groups,const int* node_to_group,const double* solved,
    float* target,unsigned* bad) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(*bad || i>=*count) return;
    if(solved[CAP]!=0 || solved[OUT+CAP]!=0) {reject(bad);return;}
    if(rows[6*i]<0) {target[i]=float(rows[6*i+3]);return;}
    unsigned node=unsigned(rows[6*i]),side=unsigned(rows[6*i+1]);int group=node_to_group[side*nodes+node];
    if(group<0 || group>=CAP) {reject(bad);return;}
    double value=solved[side*OUT+group]/groups[4*(side*nodes+node)+3];
    if(!isfinite(value) || value<0 || value>3.402823466e38) {reject(bad);return;}
    target[i]=float(value);
}
