// Frozen-contact GPU primitive; no simulator pointers or native launch adapter.
// One block solves one diagonal-plus-rank-one unilateral contact system.
#include <cuda_runtime.h>
#include <math.h>
constexpr int CAP = 128;
constexpr int IN = 3 + 4 * CAP;
constexpr int OUT = CAP + 2;
extern "C" __global__ void solve_rank_one(const double* inputs, double* outputs, int count) {
    int sample = blockIdx.x, k = threadIdx.x;
    if (sample >= count) return;
    const double* a = inputs + sample * IN;
    double* out = outputs + sample * OUT;
    __shared__ double reduce[CAP], lo, hi, previous_sum;
    __shared__ int invalid;
    if (k == 0) { invalid = 0; lo = 0.; hi = 0.; }
    __syncthreads();
    int n = (isfinite(a[0]) && a[0] >= 0 && a[0] <= CAP) ? int(a[0]) : -1;
    double r = a[1], h = a[2];
    bool valid = n >= 0 && a[0] == n && isfinite(r) && r >= 0 && isfinite(h) && h > 0;
    double d = 1., c = 0., old = 0.;
    if (valid && k < n) {
        d = a[3 + 4*k]; c = a[4 + 4*k]; old = a[5 + 4*k];
        valid = isfinite(d) && d > 0 && isfinite(c) && isfinite(old) && old >= 0;
    }
    if (!valid) atomicExch(&invalid, 1);
    reduce[k] = valid && k < n ? old : 0.;
    __syncthreads();
    for (int s = CAP/2; s; s /= 2) {
        if (k < s) reduce[k] += reduce[k+s];
        __syncthreads();
    }
    if (k == 0) previous_sum = reduce[0];
    __syncthreads();
    // Reject the entire sample before writing any impulses.
    if (invalid) { if (k == 0) out[CAP] = -1.; return; }
    // Solve for the CHANGE in shared rigid velocity. Forming a total impulse
    // offset and subtracting r*sum(old) loses precision at extreme mass ratios.
    reduce[k] = k < n ? fmax(0., -c) : 0.;
    __syncthreads();
    for (int s = CAP/2; s; s /= 2) {
        if (k < s) reduce[k] = fmax(reduce[k], reduce[k+s]);
        __syncthreads();
    }
    if (k == 0) { hi = reduce[0]; lo = -r*previous_sum; }
    __syncthreads();
    // Monotone scalar root z = r sum(max(-old, -(c+z)/d)).
    // Fixed bisection is independent of the CPU active-set reference.
    for (int iteration = 0; iteration < 80; ++iteration) {
        double middle = lo + .5*(hi-lo);
        reduce[k] = k < n ? fmax(-old, -(c+middle)/d) : 0.;
        __syncthreads();
        for (int s = CAP/2; s; s /= 2) {
            if (k < s) reduce[k] += reduce[k+s];
            __syncthreads();
        }
        if (k == 0) { if (middle < r*reduce[0]) lo = middle; else hi = middle; }
        __syncthreads();
    }
    double z = lo + .5*(hi-lo);
    out[k] = k < n ? fmax(0., old-(c+z)/d) : 0.;
    if (k == 0) { out[CAP] = 0.; out[CAP+1] = z; }
}
