// Version-pinned Cartesian coupon experiment; validates the whole batch before writes.
#pragma once
#include <vector_types.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#ifdef __CUDACC__
#define CA_HD __host__ __device__
#else
#define CA_HD
#endif
template<class T> CA_HD T ca_field(const unsigned char* p, size_t offset) {
    return *reinterpret_cast<const T*>(p+offset);
}
// Shared CPU/device validation. Return codes are evidence, never a fallback policy.
CA_HD unsigned ca_validate(const unsigned char* table, const uint4* info,
    const unsigned char* constraints, unsigned count, const unsigned char* soft,
    unsigned nodes, const unsigned char* prep, const unsigned char* core,
    const unsigned char* arti) {
    if(count>1024 || !nodes || nodes>4096 || !table || !soft) return 1;
    if(ca_field<unsigned>(soft,372)!=nodes) return 2;
    if(!count) return 0;
    if(!info || !constraints || !prep || !core || !arti) return 3;
    if(ca_field<unsigned>(arti,44)!=1 || ca_field<unsigned>(arti,248)!=4) return 4;
    auto tets=ca_field<const uint4*>(soft,80);
    auto pos=ca_field<const float4*>(soft,96);
    unsigned nt=ca_field<unsigned>(soft,376);
    if(!tets || !pos || !nt) return 5;
    unsigned char seen[4096]={}; unsigned groups[2]={};
    for(unsigned i=0;i<count;++i) {
        uint4 row=info[i]; unsigned tet=row.z&0xfffffu, body=row.z>>20;
        if(uintptr_t(table)+size_t(body)*432!=uintptr_t(soft) || tet>=nt) return 6;
        bool top=(row.y&1u)!=0;
        if((top && (row.x>=64 || row.y!=7)) || (!top && row.x!=0xffffffffu)) return 7;
        const unsigned char* block=constraints+(i/32)*3200+(i%32)*16;
        float4 n=ca_field<float4>(block,1536), bc=ca_field<float4>(block,2560);
        if(n.x!=0 || n.y!=0 || n.z!=(top?1.f:-1.f) || !isfinite(n.w)) return 8;
        float weights[4]={bc.x,bc.y,bc.z,bc.w}; int selected=-1;
        for(unsigned j=0;j<4;++j) {
            if(weights[j]==1.f && selected<0) selected=int(j);
            else if(weights[j]!=0.f) return 9;
        }
        if(selected<0) return 9;
        uint4 ids=tets[tet]; unsigned index[4]={ids.x,ids.y,ids.z,ids.w};
        unsigned node=index[selected]; if(node>=nodes) return 10;
        if(!isfinite(pos[node].w) || pos[node].w<=0) return 11;
        unsigned bit=top?2:1;
        if(seen[node] && seen[node]!=bit) return 12;
        if(!seen[node]) {seen[node]=bit; if(++groups[top?1:0]>128) return 13;}
        if(top) {
            auto indices=ca_field<const unsigned*>(prep,272);
            auto velocity=ca_field<const float4*>(core,40);
            if(!indices || !velocity || indices[row.x]!=0) return 14;
            float4 a=velocity[7], d=velocity[15];
            if(a.x!=0 || a.y!=0 || a.z!=0 || d.x!=0 || d.y!=0 || d.z!=0) return 15;
        }
    }
    return 0;
}
CA_HD unsigned ca_activate(uint4* info, unsigned count) {
    unsigned changed=0;
    for(unsigned i=0;i<count;++i) {
        changed+=!(info[i].w&0x80000000u); info[i].w|=0x80000000u;
    }
    return changed;
}
#undef CA_HD
