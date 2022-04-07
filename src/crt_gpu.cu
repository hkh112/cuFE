// CUDA libraries.
#include <cuda.h>
#include <cuda_runtime.h>

// Include associated header file.
#include "params.h"
#include "params_gpu.cuh"
#include "arith_rns.cuh"
// #include "crt_gpu.cuh"
#include <stdio.h>


__global__ void crt_mul_acc_gpu(const uint32_t *msk, uint32_t *y_crt, uint32_t *sk_y)
{
   uint64_t mac=0, i, j;
   uint32_t tid =threadIdx.x, bid = blockIdx.x*blockDim.x;
   for (i = 0; i < SIFE_L; ++i) 
   {
      for (j = 0; j < SIFE_NMODULI; ++j) {
         mac = (uint64_t)y_crt[j*SIFE_L + i]*msk[j*SIFE_N + tid + i*SIFE_NMODULI*SIFE_N + bid];
         mac = mac + sk_y[j*SIFE_N + tid + bid];
         sk_y[j*SIFE_N + tid + bid] = mod_prime_gpu(mac, j);
      }
   }
}

__global__ void crt_mul_acc_gpu2(const uint32_t *msk, uint32_t *y_crt, uint32_t *sk_y)
{
   uint64_t mac=0, i, j;
   uint32_t tid =threadIdx.x, bid = blockIdx.x*blockDim.x;
   uint32_t repeat = blockIdx.y;
   for (i = 0; i < SIFE_L; ++i) 
   {
      for (j = 0; j < SIFE_NMODULI; ++j) {
         mac = (uint64_t)y_crt[repeat*SIFE_NMODULI*SIFE_L + j*SIFE_L + i]*msk[j*SIFE_N + tid + i*SIFE_NMODULI*SIFE_N + bid];
         mac = mac + sk_y[repeat*SIFE_NMODULI*SIFE_N + j*SIFE_N + tid + bid];
         sk_y[repeat*SIFE_NMODULI*SIFE_N + j*SIFE_N + tid + bid] = mod_prime_gpu(mac, j);
      }
   }
}

__global__ void crt_convert_generic_gpu(const uint32_t *a, uint32_t *a_crt)
{
   uint32_t tid = threadIdx.x, bid = blockIdx.x;
   uint32_t repeat = blockIdx.y;
   a_crt[repeat*gridDim.x*blockDim.x + bid*blockDim.x + tid] = mod_prime_gpu(a[repeat*blockDim.x + tid], bid);
}

 extern "C" void crt_convert_generic_gpu_call(uint32_t a[SIFE_L], uint32_t a_crt[SIFE_NMODULI][SIFE_L], const int len) {

    uint32_t *d_a, *d_acrt;
    cudaMalloc((void**) &d_a, SIFE_L * sizeof(uint32_t));
    cudaMalloc((void**) &d_acrt, SIFE_NMODULI*SIFE_L * sizeof(uint32_t));

    cudaMemcpy(d_a, a, SIFE_L * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_acrt, a_crt, SIFE_NMODULI*SIFE_L * sizeof(uint32_t), cudaMemcpyHostToDevice);

    crt_convert_generic_gpu<<<SIFE_NMODULI, len>>>(d_a, d_acrt);
    cudaMemcpy(a_crt, d_acrt, SIFE_NMODULI*SIFE_L * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_acrt);
}

__global__ void crt_mxm_gpu(uint32_t *a)
{
   uint64_t mxm;
   uint32_t tid = threadIdx.x, bid = blockIdx.x;
   uint32_t repeat = blockIdx.y;

   mxm = (uint64_t)a[repeat*gridDim.x*blockDim.x + bid*SIFE_L + tid] * SIFE_SCALE_M_MOD_Q_I_GPU[bid];
   a[repeat*gridDim.x*blockDim.x + bid*SIFE_L + tid] = mod_prime_gpu(mxm, bid);
}



 extern "C" void crt_mxm_gpu_call(uint32_t a_crt[SIFE_NMODULI][SIFE_L]) {

    uint32_t *d_acrt;
    cudaMalloc((void**) &d_acrt, SIFE_NMODULI*SIFE_L * sizeof(uint32_t));
    cudaMemcpy(d_acrt, a_crt, SIFE_NMODULI*SIFE_L * sizeof(uint32_t), cudaMemcpyHostToDevice);
    crt_mxm_gpu<<<SIFE_NMODULI, SIFE_L>>>(d_acrt);
    cudaMemcpy(a_crt, d_acrt, SIFE_NMODULI*SIFE_L * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_acrt);
}


