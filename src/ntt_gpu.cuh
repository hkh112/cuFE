#include <stdint.h>
#include "params.h"

__device__ uint32_t add_mod_ntt_gpu(uint32_t a, uint32_t b, uint32_t sel);
__device__ uint32_t sub_mod_ntt_gpu(uint32_t a, uint32_t b, uint32_t sel);

__global__ void point_add_mod_gpu(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b);
__global__ void point_add_mod_gpu2(uint32_t *d_c, uint32_t *d_m);
__global__ void point_add_mod_gpu3(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b);

__global__ void point_mul_gpu(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b);
__global__ void point_mul_gpu2(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b);
__global__ void point_mul_gpu3(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b);

__global__ void CT_forward_gpu_1block_3round(uint32_t a[SIFE_N]);
__global__ void GS_reverse_gpu_1block_3round(uint32_t* a);

__global__ void keygen_gpu(const uint32_t *y, uint32_t *d_msk, uint32_t *d_sky);
__global__ void decryption_gpu(const uint32_t *y, uint32_t *c, uint32_t* d_sky, uint32_t *dev_dy);



