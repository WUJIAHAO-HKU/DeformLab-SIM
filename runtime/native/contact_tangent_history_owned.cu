// Candidate-owned row buffers only. Does not intercept any simulator kernel.
// Device execution and host/device agreement require an explicit later GPU proof.
#include "contact_tangent_history.h"
extern "C" __global__ void contact_tangent_history_owned(
    const contact_tangent::Input* inputs, contact_tangent::Output* outputs,
    unsigned* status, unsigned rows) {
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=rows) return;
  contact_tangent::Output candidate;
  bool valid=contact_tangent::project(inputs[i],candidate);
  if (valid) outputs[i]=candidate;
  status[i]=valid?1u:0u;
}
