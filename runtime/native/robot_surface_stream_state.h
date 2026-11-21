#pragma once
#include <stdint.h>
struct SurfaceStreamState {
 uint64_t input[9];
 uint64_t header,cache,seen,active,boundary,plan,selected,zero;
 uint64_t info,sid,constraints,lambda,rigid,count;
 uint64_t fixed_topology,frame_info,frame_ids,guards,telemetry;
 uint64_t native_binding[6];
 uint32_t native_validation,map_count;
 uint32_t nodes,tets,boundary_count,capacity,native_count,guard_count,frame,reserved;
 double cx,cy,hx,hy,plane,epsilon;
};
