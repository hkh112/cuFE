#include <stdint.h>
#include "params.h"

extern "C" void point_gpu_poly_add_mod(uint32_t a[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t *b, uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t e_crt[SIFE_L][SIFE_NMODULI][SIFE_N]);
extern "C" void CT_forward_Moduli_GPU(uint32_t a[SIFE_NMODULI][SIFE_N]);
extern "C" void CT_forward_Moduli_L_GPU(uint32_t *a);
__global__ void point_add_mod_gpu(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b);
__global__ void point_mul_gpu(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b);
__global__ void point_mul_gpu2(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b);
__global__ void CT_forward_gpu_1block_1round(uint32_t a[SIFE_N]);
__global__ void GS_reverse_gpu_1block_1round(uint32_t a[SIFE_N]) ;
__global__ void point_add_mod_gpu2(uint32_t *d_c, uint32_t *d_m);
__global__ void point_add_mod_gpu3(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b);
__global__ void poly_sub_mod_gpu(uint32_t *a, uint32_t *b, uint32_t *c);
__global__ void CT_forward_gpu_1block_2round(uint32_t a[SIFE_N]);
__global__ void GS_reverse_gpu_1block_2round(uint32_t a[SIFE_N]);



__device__ uint32_t add_mod_ntt_gpu(uint32_t a, uint32_t b, uint32_t sel);
__device__ uint32_t sub_mod_ntt_gpu(uint32_t a, uint32_t b, uint32_t sel);

__global__ void CT_forward_gpu_2block_1kernel_1round_batch(uint32_t* a);
__global__ void CT_forward_gpu_16block_2kernel_1round_1_batch(uint32_t* a);
__global__ void CT_forward_gpu_16block_2kernel_1round_2_batch(uint32_t* a);
__global__ void CT_forward_gpu_64block_2kernel_1round_1_batch(uint32_t* a);
__global__ void CT_forward_gpu_64block_2kernel_1round_2_batch(uint32_t* a);
__global__ void GS_reverse_gpu_64block_2kernel_1round_1_batch(uint32_t* a);
__global__ void GS_reverse_gpu_64block_2kernel_1round_2_batch(uint32_t* a);
__global__ void GS_reverse_gpu_64block_2kernel_1round_1_batch2(uint32_t* a);
__global__ void GS_reverse_gpu_64block_2kernel_1round_2_batch2(uint32_t* a);
__global__ void GS_reverse_gpu_64block_2kernel_1round_1_batch3(uint32_t* a);
__global__ void GS_reverse_gpu_64block_2kernel_1round_2_batch3(uint32_t* a);

__global__ void point_mul_gpu_batch(uint32_t a[SIFE_N], uint32_t b[SIFE_N], uint32_t c[SIFE_N]);
__global__ void poly_add_mod_gpu_batch(uint32_t a[SIFE_N], uint32_t b[SIFE_N], uint32_t c[SIFE_N]);
__global__ void poly_sub_mod_gpu_batch(uint32_t a[SIFE_N], uint32_t b[SIFE_N], uint32_t c[SIFE_N]);



__global__ void keygen_gpu(const uint32_t *y, uint32_t *d_msk, uint32_t *d_sky);
__global__ void decryption_gpu(const uint32_t *y, uint32_t *c, uint32_t* d_sky, uint32_t *dev_dy);

__global__ void CT_forward_gpu_1block_3round(uint32_t a[SIFE_N]);
__global__ void GS_reverse_gpu_1block_3round(uint32_t a[SIFE_N]);
