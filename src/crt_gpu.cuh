#include <stdint.h>
#include "params.h"
void crt_convert_generic_gpu_call(uint32_t a[SIFE_L], uint32_t a_crt[SIFE_NMODULI][SIFE_L], const int len);

__global__ void crt_convert_generic_gpu(const uint32_t *a, uint32_t *a_crt);
__global__ void crt_mxm_gpu(uint32_t *a);
__global__ void crt_mul_acc_gpu(const uint32_t *msk, uint32_t *y_crt, uint32_t *sk_y);



__global__ void crt_mul_acc_gpu2(const uint32_t *msk, uint32_t *y_crt, uint32_t *sk_y);