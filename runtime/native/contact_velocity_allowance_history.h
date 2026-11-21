// Updated-gap velocity allowance for a frozen row with one paired impulse state.
// G'=gap/h + response; V'=vn + response. Enforce G'>=0 and V'+G'>=0.
// Equivalent complementarity residual: min(G, (G+V)/2) + response.
// This specifies a discrete contact law, not time-of-impact integration.
#pragma once
#include "contact_normal_history.h"
#ifdef __CUDACC__
#define CV_HD __host__ __device__
#else
#define CV_HD
#endif
namespace contact_velocity_allowance {
struct Input {
  double current_gap_m, current_normal_velocity, old_impulse;
  double mobility_majorizer, substep_s;
};
using Output=contact_normal::Output;
CV_HD inline bool project(const Input& in,Output& out) {
  if (!isfinite(in.current_gap_m)||!isfinite(in.current_normal_velocity)||
      !isfinite(in.substep_s)||!(in.substep_s>0)) return false;
  double G=in.current_gap_m/in.substep_s;
  double updated=fmin(G,.5*G+.5*in.current_normal_velocity);
  double gap=in.substep_s*updated;
  if (!isfinite(G)||!isfinite(updated)||!isfinite(gap)) return false;
  contact_normal::Input normal={gap,in.old_impulse,in.mobility_majorizer,in.substep_s};
  return contact_normal::project(normal,out);
}
}
#undef CV_HD
