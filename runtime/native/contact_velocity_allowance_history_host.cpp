#include "contact_velocity_allowance_history.h"
extern "C" int contact_velocity_allowance_history_host(
    const double* input,double* output) {
  if (!input||!output) return 0;
  contact_velocity_allowance::Input in={input[0],input[1],input[2],input[3],input[4]};
  contact_velocity_allowance::Output out;
  if (!contact_velocity_allowance::project(in,out)) return 0;
  output[0]=out.impulse;output[1]=out.delta_impulse;
  return 1;
}
