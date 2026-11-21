// Candidate-owned CUDA buffers. No native binding or verified device execution.
#include "contact_velocity_allowance_history.h"
extern "C" __global__ void contact_velocity_allowance_history_owned(
    const contact_velocity_allowance::Input* inputs,
    contact_velocity_allowance::Output* outputs,unsigned* status,unsigned rows) {
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=rows)return;
  contact_velocity_allowance::Output candidate;
  bool valid=contact_velocity_allowance::project(inputs[i],candidate);
  if(valid)outputs[i]=candidate;
  status[i]=valid?1u:0u;
}
