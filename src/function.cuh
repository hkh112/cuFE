

__device__ uint32_t add_mod_ntt_gpu_2(uint32_t a, uint32_t b, uint32_t sel);
__device__ uint32_t sub_mod_ntt_gpu_2(uint32_t a, uint32_t b, uint32_t sel);
__global__ void point_mul_gpu_2(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b);
__global__ void point_mul_gpu2_2(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b);
__global__ void point_mul_gpu2_3(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b);
__global__ void point_add_mod_gpu2_2(uint32_t *d_c, uint32_t *d_m);
__global__ void point_add_mod_gpu3_2(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b);
__global__ void poly_sub_mod_gpu2(uint32_t *a, uint32_t *b, uint32_t *c);
__global__ void gaussian_sampler_S3_gpu2(uint8_t *rk, uint32_t *d_c);
__global__ void GS_reverse_gpu_1block_2round2(uint32_t a[SIFE_N]);
__global__ void GS_reverse_gpu_1block_3round2(uint32_t a[SIFE_N]);