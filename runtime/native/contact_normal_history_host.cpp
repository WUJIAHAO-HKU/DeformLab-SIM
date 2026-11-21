#include "contact_normal_history.h"
extern "C" int contact_normal_history_host(const double* input,double* output) {
  if (!input||!output) return 0;
  contact_normal::Input in={input[0],input[1],input[2],input[3]};
  contact_normal::Output out;
  if (!contact_normal::project(in,out)) return 0;
  output[0]=out.impulse;
  output[1]=out.delta_impulse;
  return 1;
}
