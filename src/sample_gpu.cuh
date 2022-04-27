#include <stdint.h>
#include "params.h"

__global__ void gaussian_sampler_S1_gpu(uint8_t *rk, uint32_t *sample);
__global__ void gaussian_sampler_S2_gpu(uint8_t *rk, uint32_t *sample);

__device__ uint32_t add_mod_ntt_gpu_2(uint32_t a, uint32_t b, uint32_t sel);
__global__ void gaussian_sampler_S3_gpu(uint8_t *rk, uint32_t *d_c);