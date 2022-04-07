#include <stdint.h>
#include "params.h"

// int gaussian_S1_gpu(unsigned char *seed, uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N]);
int gaussian_S1_gpu2(unsigned char *seed, uint32_t *msk);
__global__ void gaussian_sampler_S1_gpu(uint8_t *rk, uint32_t *sample);
__global__ void gaussian_sampler_S2_gpu(uint8_t *rk, uint32_t *sample);
__global__ void gaussian_sampler_S3_gpu(uint8_t *rk, uint32_t *sample);