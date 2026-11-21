// Owned-buffer normal row primitive; native and GPU execution not verified.
#include "contact_normal_history.h"
extern "C" __global__ void contact_normal_history_owned(
    const contact_normal::Input* inputs,contact_normal::Output* outputs,
    unsigned* status,unsigned rows) {
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=rows) return;
  contact_normal::Output candidate;
  bool valid=contact_normal::project(inputs[i],candidate);
  if (valid) outputs[i]=candidate;
  status[i]=valid?1u:0u;
}
