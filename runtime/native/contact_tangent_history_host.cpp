#include "contact_tangent_history.h"
extern "C" int contact_tangent_history_host(const double* input, double* output) {
  if (!input||!output) return 0;
  contact_tangent::Input in={
    {input[0],input[1],input[2]}, {input[3],input[4],input[5]},
    {input[6],input[7],input[8]}, input[9],input[10],input[11]};
  contact_tangent::Output out;
  if (!contact_tangent::project(in,out)) return 0;
  const double result[]={out.impulse.x,out.impulse.y,out.impulse.z,
    out.delta_impulse.x,out.delta_impulse.y,out.delta_impulse.z};
  for (int i=0;i<6;++i) output[i]=result[i];
  return 1;
}
