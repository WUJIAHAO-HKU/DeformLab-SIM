// Frozen normal complementarity update in impulse units; no runtime binding.
#pragma once
#include "contact_tangent_history.h"
#ifdef __CUDACC__
#define CN_HD __host__ __device__
#else
#define CN_HD
#endif
namespace contact_normal {
struct Input {
  double current_gap_m, old_impulse, mobility_majorizer, substep_s;
};
struct Output { double impulse, delta_impulse; };
CN_HD inline bool project(const Input& in, Output& out) {
  if (!isfinite(in.current_gap_m)||!isfinite(in.old_impulse)||in.old_impulse<0||
      !isfinite(in.mobility_majorizer)||!(in.mobility_majorizer>0)||
      !isfinite(in.substep_s)||!(in.substep_s>0)) return false;
  double trial=in.old_impulse-in.current_gap_m/in.substep_s/in.mobility_majorizer;
  if (!isfinite(trial)) return false;
  double next=fmax(0.,trial), delta=next-in.old_impulse;
  if (!isfinite(delta)) return false;
  out={next,delta};
  return true;
}
}
#undef CN_HD
