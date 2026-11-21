#pragma once
#include <array>
#include <vector>
#include <cstddef>
// Two existing actors, one shared allocation. Failure leaves lists unchanged.
// Never invent an actor when the native narrowphase supplied no pair.
inline bool rebalance_contact_slots(std::array<std::vector<unsigned>,2>& slots,
        const std::array<std::size_t,2>& required,std::array<std::size_t,2>& borrowed){
    borrowed={0,0};
    if(slots[0].size()+slots[1].size()<required[0]+required[1])return false;
    for(unsigned side=0;side<2;++side){
        if(required[side] && slots[side].empty())return false;
    }
    for(unsigned side=0;side<2;++side){
        if(slots[side].size()>=required[side])continue;
        auto& dst=slots[side];auto& src=slots[1-side];
        std::size_t count=required[side]-dst.size();
        dst.insert(dst.end(),src.end()-count,src.end());src.resize(src.size()-count);borrowed[side]=count;
    }
    return true;
}
