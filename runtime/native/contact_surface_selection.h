#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <set>
#include <utility>
#include <vector>

// Used only by the opt-in band-complete Cartesian contact experiment.
// Preserve full-face order when capacity permits. Under scarce capacity,
// cover every node within the supplied geometric band and fill by distance.
// This is an instantaneous geometric guarantee, not a time-of-impact bound.
// Failure leaves output unchanged. The caller must verify actor ownership,
// moving-plane geometry and omitted-node feasibility throughout integration.
inline bool select_contact_surface_nodes(const std::vector<unsigned>& nodes,
        const std::vector<double>& gaps,std::size_t capacity,double band,
        std::vector<unsigned>& selected){
    if(nodes.size()!=gaps.size() || !std::isfinite(band) || band<0)return false;
    if(std::set<unsigned>(nodes.begin(),nodes.end()).size()!=nodes.size())return false;
    std::size_t required=0;
    for(double g:gaps){if(!std::isfinite(g))return false;required+=g<=band;}
    if(required>capacity)return false;
    if(capacity>=nodes.size()){selected=nodes;return true;}
    std::vector<std::pair<double,unsigned>> ranked;
    for(std::size_t i=0;i<nodes.size();++i)ranked.emplace_back(gaps[i],nodes[i]);
    std::sort(ranked.begin(),ranked.end());
    std::vector<unsigned> result;result.reserve(capacity);
    for(std::size_t i=0;i<capacity;++i)result.push_back(ranked[i].second);
    selected=std::move(result);return true;
}
