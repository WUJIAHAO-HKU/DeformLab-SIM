// Frozen-row cumulative impulse projection, shared by host and CUDA wrappers.
// No native integration is implied. Normal impulse and geometry are fixed inputs.
// The caller must include old_impulse in current_relative_velocity exactly once.
// On failure output is unchanged. Impulses use kg*m/s, not position multipliers.
#pragma once
#include <math.h>
#ifdef __CUDACC__
#define CT_HD __host__ __device__
#else
#define CT_HD
#endif
namespace contact_tangent {
struct Vec { double x, y, z; };
struct Input {
  Vec normal, current_relative_velocity, old_impulse;
  double mobility_majorizer, friction, normal_impulse;
};
struct Output { Vec impulse, delta_impulse; };
CT_HD inline Vec add(Vec a, Vec b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
CT_HD inline Vec sub(Vec a, Vec b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
CT_HD inline Vec mul(Vec a, double s) { return {a.x*s,a.y*s,a.z*s}; }
CT_HD inline double dot(Vec a, Vec b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
CT_HD inline double norm(Vec a) { return hypot(hypot(a.x,a.y),a.z); }
CT_HD inline bool finite(Vec a) {
  return isfinite(a.x)&&isfinite(a.y)&&isfinite(a.z);
}
CT_HD inline bool project(const Input& in, Output& out) {
  if (!finite(in.normal)||!finite(in.current_relative_velocity)||
      !finite(in.old_impulse)||!isfinite(in.mobility_majorizer)||
      !(in.mobility_majorizer>0)||!isfinite(in.friction)||in.friction<0||
      !isfinite(in.normal_impulse)||in.normal_impulse<0) return false;
  const double normal_norm=norm(in.normal), old_norm=norm(in.old_impulse);
  if (fabs(normal_norm-1)>1e-12 ||
      fabs(dot(in.normal,in.old_impulse))>1e-12*fmax(old_norm,1e-30)) return false;
  const double radius=in.friction*in.normal_impulse;
  if (!isfinite(radius)) return false;
  Vec slip=sub(in.current_relative_velocity,
               mul(in.normal,dot(in.normal,in.current_relative_velocity)));
  Vec trial=sub(in.old_impulse,mul(slip,1/in.mobility_majorizer));
  // Remove roundoff normal leakage before disk projection.
  trial=sub(trial,mul(in.normal,dot(trial,in.normal)));
  double length=norm(trial);
  if (!finite(trial)||!isfinite(length)) return false;
  Vec next=radius==0?Vec{0,0,0}:
      length>radius?mul(trial,radius/length):trial;
  Vec delta=sub(next,in.old_impulse);
  if (!finite(next)||!finite(delta)) return false;
  out={next,delta};
  return true;
}
}
#undef CT_HD
