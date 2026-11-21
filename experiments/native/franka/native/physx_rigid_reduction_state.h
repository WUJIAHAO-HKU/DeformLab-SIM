#pragma once
#include <vector_types.h>
struct RigidReductionState {
 double3 residual,increment,native_roundoff_sum,continuity_error;
 float4 before,published;
 unsigned rigid_node,pending,initialized,faults;
 unsigned long long samples;
};
static_assert(sizeof(RigidReductionState)==160,"Pinned observation layout");
