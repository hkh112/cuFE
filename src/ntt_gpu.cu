// #include "ntt_gpu.cuh"

#include <stdint.h>
#include "params.h"
#include "params_gpu.cuh"
#include "arith_rns.cuh"
#include "ntt_gpu.cuh"
#include "consts.cuh"
#include <stdio.h>


__device__ uint32_t add_mod_ntt_gpu(uint32_t a, uint32_t b, uint32_t sel)
{
	uint64_t c;

	c = (uint64_t)a + (uint64_t)b;

	if (c >= SIFE_MOD_Q_I_gpu[sel]) {
		c -= SIFE_MOD_Q_I_gpu[sel];
	}
	return (uint32_t)c;
}

__device__ uint32_t sub_mod_ntt_gpu(uint32_t a, uint32_t b, uint32_t sel)//returns a-b Mod Q
{
	uint64_t c;

	c = (uint64_t)a + (uint64_t)SIFE_MOD_Q_I_gpu[sel] - (uint64_t)b;

	if (c >= SIFE_MOD_Q_I_gpu[sel]) {
		c -= SIFE_MOD_Q_I_gpu[sel];
	}
	return (uint32_t)c;
}

__global__ void point_add_mod_gpu(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b)
{
	uint32_t tid = threadIdx.x, bid = blockIdx.x*SIFE_NMODULI*SIFE_N;
	int32_t i, j;
	for (j = 0; j < SIFE_NMODULI; ++j)
	{
		for (i = 0; i < SIFE_N/1024; ++i)
		{
			d_c[bid+tid+j*SIFE_N+i*1024] = add_mod_ntt_gpu(d_a[bid+tid+j*SIFE_N+i*1024], d_b[bid+tid+j*SIFE_N+i*1024], j);
		}
	}	
}

__global__ void point_add_mod_gpu2(uint32_t *d_c, uint32_t *d_m)
{
	uint32_t tid = threadIdx.x, bid = blockIdx.x*SIFE_NMODULI*SIFE_N, bid2 = blockIdx.x;
	uint32_t repeat = blockIdx.y;
	int32_t j, i;
	for (j = 0; j < SIFE_NMODULI; ++j)
	{
		for (i = 0; i < SIFE_N/1024; ++i)
		{
			d_c[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + bid+tid+j*SIFE_N+i*+1024] = mod_prime_gpu(d_c[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + bid+tid+j*SIFE_N+i*+1024] + d_m[repeat*SIFE_L*SIFE_NMODULI + bid2 + j*SIFE_L], j);
		}
	}	
}

__global__ void point_add_mod_gpu3(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b)
{
	uint32_t tid = threadIdx.x, bid = blockIdx.x*SIFE_N;
	uint32_t repeat = blockIdx.y;
	int32_t i;
	for (i = 0; i < SIFE_N/1024; ++i)
	{
		d_c[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + bid+tid+i*1024] = add_mod_ntt_gpu(d_a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + bid+tid+i*1024], d_b[repeat*SIFE_NMODULI*SIFE_N + bid+tid+i*1024], blockIdx.x);
	}	
}

__global__ void point_mul_gpu(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b)
{
	uint32_t tid = threadIdx.x, bid = blockIdx.x*SIFE_NMODULI*SIFE_N;
	int32_t i, j;
	for (j = 0; j < SIFE_NMODULI; ++j)
	{
		for (i = 0; i < SIFE_N/1024; ++i)
		{
			d_c[bid+tid+j*SIFE_N+i*1024] = mul_mod_ntt_gpu(d_a[tid+j*SIFE_N+i*1024], d_b[bid+tid+j*SIFE_N+i*1024], j);
		}		
	}	
}

__global__ void point_mul_gpu2(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b)
{
	uint32_t tid = threadIdx.x, bid = blockIdx.x*SIFE_NMODULI*SIFE_N;
	uint32_t repeat = blockIdx.y;
	int32_t i, j;
	for (j = 0; j < SIFE_NMODULI; ++j)
	{
		for (i = 0; i < SIFE_N/1024; ++i)
		{
			d_c[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + bid+tid+j*SIFE_N+i*1024] = mul_mod_ntt_gpu(d_a[repeat*SIFE_NMODULI*SIFE_N + tid+j*SIFE_N+i*1024], d_b[bid+tid+j*SIFE_N+i*1024], j);
		}		
	}	
}

__global__ void point_mul_gpu3(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b)
{
	uint32_t tid = threadIdx.x, bid = blockIdx.x*SIFE_N;
	uint32_t repeat = blockIdx.y;
	int32_t i;
	for (i = 0; i < SIFE_N/1024; ++i)
	{
		d_c[repeat*(SIFE_L+1)*gridDim.x*SIFE_N + bid+tid+i*1024] = mul_mod_ntt_gpu(d_a[repeat*gridDim.x*SIFE_N + bid + tid+i*1024], d_b[bid+tid+i*1024], blockIdx.x);			
	}	
}

__global__ void CT_forward_gpu_1block_3round(uint32_t* a) {
	int64_t t, S, V, g0, g1, g2, g3, g4, g5, g6, g7;
	uint32_t thread_id, operation_id;
	__shared__ uint32_t s_a[SIFE_N];
	uint32_t bid = blockIdx.x%SIFE_NMODULI;
	uint32_t repeat = blockIdx.y;

	t = SIFE_N;
	thread_id = threadIdx.x;
	s_a[thread_id] = a[repeat*gridDim.x*SIFE_N + thread_id + blockIdx.x * SIFE_N];
	s_a[thread_id + 512] = a[repeat*gridDim.x*SIFE_N + thread_id + 512 + blockIdx.x * SIFE_N];
	s_a[thread_id + 1024] = a[repeat*gridDim.x*SIFE_N + thread_id + 1024 + blockIdx.x * SIFE_N];
	s_a[thread_id + 1536] = a[repeat*gridDim.x*SIFE_N + thread_id + 1536 + blockIdx.x * SIFE_N];
	s_a[thread_id + 2048] = a[repeat*gridDim.x*SIFE_N + thread_id + 2048 + blockIdx.x * SIFE_N];
	s_a[thread_id + 2560] = a[repeat*gridDim.x*SIFE_N + thread_id + 2560 + blockIdx.x * SIFE_N];
	s_a[thread_id + 3072] = a[repeat*gridDim.x*SIFE_N + thread_id + 3072 + blockIdx.x * SIFE_N];
	s_a[thread_id + 3584] = a[repeat*gridDim.x*SIFE_N + thread_id + 3584 + blockIdx.x * SIFE_N];

	t = t / 8;
	operation_id = (threadIdx.x / t) * 8 * t + threadIdx.x % t;
	S = psi_gpu[bid][SIFE_N / t / 8 + operation_id / t / 8];
	V = mul_mod_ntt_gpu(s_a[operation_id + t * 4], S, bid);
	g4 = sub_mod_ntt_gpu(s_a[operation_id], V, bid);
	g0 = add_mod_ntt_gpu(s_a[operation_id], V, bid);
	V = mul_mod_ntt_gpu(s_a[operation_id + t * 5], S, bid);
	g5 = sub_mod_ntt_gpu(s_a[operation_id + t], V, bid);
	g1 = add_mod_ntt_gpu(s_a[operation_id + t], V, bid);
	V = mul_mod_ntt_gpu(s_a[operation_id + t * 6], S, bid);
	g6 = sub_mod_ntt_gpu(s_a[operation_id + t * 2], V, bid);
	g2 = add_mod_ntt_gpu(s_a[operation_id + t * 2], V, bid);
	V = mul_mod_ntt_gpu(s_a[operation_id + t * 7], S, bid);
	g7 = sub_mod_ntt_gpu(s_a[operation_id + t * 3], V, bid);
	g3 = add_mod_ntt_gpu(s_a[operation_id + t * 3], V, bid);
	S = psi_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4];
	V = mul_mod_ntt_gpu(g2, S, bid);
	g2 = sub_mod_ntt_gpu(g0, V, bid);
	g0 = add_mod_ntt_gpu(g0, V, bid);
	V = mul_mod_ntt_gpu(g3, S, bid);
	g3 = sub_mod_ntt_gpu(g1, V, bid);
	g1 = add_mod_ntt_gpu(g1, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4 + 1];
	V = mul_mod_ntt_gpu(g6, S, bid);
	g6 = sub_mod_ntt_gpu(g4, V, bid);
	g4 = add_mod_ntt_gpu(g4, V, bid);
	V = mul_mod_ntt_gpu(g7, S, bid);
	g7 = sub_mod_ntt_gpu(g5, V, bid);
	g5 = add_mod_ntt_gpu(g5, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	V = mul_mod_ntt_gpu(g1, S, bid);
	s_a[operation_id + t] = sub_mod_ntt_gpu(g0, V, bid);
	s_a[operation_id] = add_mod_ntt_gpu(g0, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 1];
	V = mul_mod_ntt_gpu(g3, S, bid);
	s_a[operation_id + t * 3] = sub_mod_ntt_gpu(g2, V, bid);
	s_a[operation_id + t * 2] = add_mod_ntt_gpu(g2, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 2];
	V = mul_mod_ntt_gpu(g5, S, bid);
	s_a[operation_id + t * 5] = sub_mod_ntt_gpu(g4, V, bid);
	s_a[operation_id + t * 4] = add_mod_ntt_gpu(g4, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 3];
	V = mul_mod_ntt_gpu(g7, S, bid);
	s_a[operation_id + t * 7] = sub_mod_ntt_gpu(g6, V, bid);
	s_a[operation_id + t * 6] = add_mod_ntt_gpu(g6, V, bid);
	__syncthreads();

	t = t / 8;
	operation_id = (threadIdx.x / t) * 8 * t + threadIdx.x % t;
	S = psi_gpu[bid][SIFE_N / t / 8 + operation_id / t / 8];
	V = mul_mod_ntt_gpu(s_a[operation_id + t * 4], S, bid);
	g4 = sub_mod_ntt_gpu(s_a[operation_id], V, bid);
	g0 = add_mod_ntt_gpu(s_a[operation_id], V, bid);
	V = mul_mod_ntt_gpu(s_a[operation_id + t * 5], S, bid);
	g5 = sub_mod_ntt_gpu(s_a[operation_id + t], V, bid);
	g1 = add_mod_ntt_gpu(s_a[operation_id + t], V, bid);
	V = mul_mod_ntt_gpu(s_a[operation_id + t * 6], S, bid);
	g6 = sub_mod_ntt_gpu(s_a[operation_id + t * 2], V, bid);
	g2 = add_mod_ntt_gpu(s_a[operation_id + t * 2], V, bid);
	V = mul_mod_ntt_gpu(s_a[operation_id + t * 7], S, bid);
	g7 = sub_mod_ntt_gpu(s_a[operation_id + t * 3], V, bid);
	g3 = add_mod_ntt_gpu(s_a[operation_id + t * 3], V, bid);
	S = psi_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4];
	V = mul_mod_ntt_gpu(g2, S, bid);
	g2 = sub_mod_ntt_gpu(g0, V, bid);
	g0 = add_mod_ntt_gpu(g0, V, bid);
	V = mul_mod_ntt_gpu(g3, S, bid);
	g3 = sub_mod_ntt_gpu(g1, V, bid);
	g1 = add_mod_ntt_gpu(g1, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4 + 1];
	V = mul_mod_ntt_gpu(g6, S, bid);
	g6 = sub_mod_ntt_gpu(g4, V, bid);
	g4 = add_mod_ntt_gpu(g4, V, bid);
	V = mul_mod_ntt_gpu(g7, S, bid);
	g7 = sub_mod_ntt_gpu(g5, V, bid);
	g5 = add_mod_ntt_gpu(g5, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	V = mul_mod_ntt_gpu(g1, S, bid);
	s_a[operation_id + t] = sub_mod_ntt_gpu(g0, V, bid);
	s_a[operation_id] = add_mod_ntt_gpu(g0, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 1];
	V = mul_mod_ntt_gpu(g3, S, bid);
	s_a[operation_id + t * 3] = sub_mod_ntt_gpu(g2, V, bid);
	s_a[operation_id + t * 2] = add_mod_ntt_gpu(g2, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 2];
	V = mul_mod_ntt_gpu(g5, S, bid);
	s_a[operation_id + t * 5] = sub_mod_ntt_gpu(g4, V, bid);
	s_a[operation_id + t * 4] = add_mod_ntt_gpu(g4, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 3];
	V = mul_mod_ntt_gpu(g7, S, bid);
	s_a[operation_id + t * 7] = sub_mod_ntt_gpu(g6, V, bid);
	s_a[operation_id + t * 6] = add_mod_ntt_gpu(g6, V, bid);
	__syncthreads();

	t = t / 8;
	operation_id = (threadIdx.x / t) * 8 * t + threadIdx.x % t;
	S = psi_gpu[bid][SIFE_N / t / 8 + operation_id / t / 8];
	V = mul_mod_ntt_gpu(s_a[operation_id + t * 4], S, bid);
	g4 = sub_mod_ntt_gpu(s_a[operation_id], V, bid);
	g0 = add_mod_ntt_gpu(s_a[operation_id], V, bid);
	V = mul_mod_ntt_gpu(s_a[operation_id + t * 5], S, bid);
	g5 = sub_mod_ntt_gpu(s_a[operation_id + t], V, bid);
	g1 = add_mod_ntt_gpu(s_a[operation_id + t], V, bid);
	V = mul_mod_ntt_gpu(s_a[operation_id + t * 6], S, bid);
	g6 = sub_mod_ntt_gpu(s_a[operation_id + t * 2], V, bid);
	g2 = add_mod_ntt_gpu(s_a[operation_id + t * 2], V, bid);
	V = mul_mod_ntt_gpu(s_a[operation_id + t * 7], S, bid);
	g7 = sub_mod_ntt_gpu(s_a[operation_id + t * 3], V, bid);
	g3 = add_mod_ntt_gpu(s_a[operation_id + t * 3], V, bid);
	S = psi_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4];
	V = mul_mod_ntt_gpu(g2, S, bid);
	g2 = sub_mod_ntt_gpu(g0, V, bid);
	g0 = add_mod_ntt_gpu(g0, V, bid);
	V = mul_mod_ntt_gpu(g3, S, bid);
	g3 = sub_mod_ntt_gpu(g1, V, bid);
	g1 = add_mod_ntt_gpu(g1, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4 + 1];
	V = mul_mod_ntt_gpu(g6, S, bid);
	g6 = sub_mod_ntt_gpu(g4, V, bid);
	g4 = add_mod_ntt_gpu(g4, V, bid);
	V = mul_mod_ntt_gpu(g7, S, bid);
	g7 = sub_mod_ntt_gpu(g5, V, bid);
	g5 = add_mod_ntt_gpu(g5, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	V = mul_mod_ntt_gpu(g1, S, bid);
	s_a[operation_id + t] = sub_mod_ntt_gpu(g0, V, bid);
	s_a[operation_id] = add_mod_ntt_gpu(g0, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 1];
	V = mul_mod_ntt_gpu(g3, S, bid);
	s_a[operation_id + t * 3] = sub_mod_ntt_gpu(g2, V, bid);
	s_a[operation_id + t * 2] = add_mod_ntt_gpu(g2, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 2];
	V = mul_mod_ntt_gpu(g5, S, bid);
	s_a[operation_id + t * 5] = sub_mod_ntt_gpu(g4, V, bid);
	s_a[operation_id + t * 4] = add_mod_ntt_gpu(g4, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 3];
	V = mul_mod_ntt_gpu(g7, S, bid);
	s_a[operation_id + t * 7] = sub_mod_ntt_gpu(g6, V, bid);
	s_a[operation_id + t * 6] = add_mod_ntt_gpu(g6, V, bid);
	__syncthreads();

	t = t / 8;
	operation_id = (threadIdx.x / t) * 8 * t + threadIdx.x % t;
	S = psi_gpu[bid][SIFE_N / t / 8 + operation_id / t / 8];
	V = mul_mod_ntt_gpu(s_a[operation_id + t * 4], S, bid);
	g4 = sub_mod_ntt_gpu(s_a[operation_id], V, bid);
	g0 = add_mod_ntt_gpu(s_a[operation_id], V, bid);
	V = mul_mod_ntt_gpu(s_a[operation_id + t * 5], S, bid);
	g5 = sub_mod_ntt_gpu(s_a[operation_id + t], V, bid);
	g1 = add_mod_ntt_gpu(s_a[operation_id + t], V, bid);
	V = mul_mod_ntt_gpu(s_a[operation_id + t * 6], S, bid);
	g6 = sub_mod_ntt_gpu(s_a[operation_id + t * 2], V, bid);
	g2 = add_mod_ntt_gpu(s_a[operation_id + t * 2], V, bid);
	V = mul_mod_ntt_gpu(s_a[operation_id + t * 7], S, bid);
	g7 = sub_mod_ntt_gpu(s_a[operation_id + t * 3], V, bid);
	g3 = add_mod_ntt_gpu(s_a[operation_id + t * 3], V, bid);
	S = psi_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4];
	V = mul_mod_ntt_gpu(g2, S, bid);
	g2 = sub_mod_ntt_gpu(g0, V, bid);
	g0 = add_mod_ntt_gpu(g0, V, bid);
	V = mul_mod_ntt_gpu(g3, S, bid);
	g3 = sub_mod_ntt_gpu(g1, V, bid);
	g1 = add_mod_ntt_gpu(g1, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4 + 1];
	V = mul_mod_ntt_gpu(g6, S, bid);
	g6 = sub_mod_ntt_gpu(g4, V, bid);
	g4 = add_mod_ntt_gpu(g4, V, bid);
	V = mul_mod_ntt_gpu(g7, S, bid);
	g7 = sub_mod_ntt_gpu(g5, V, bid);
	g5 = add_mod_ntt_gpu(g5, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	V = mul_mod_ntt_gpu(g1, S, bid);
	s_a[operation_id + t] = sub_mod_ntt_gpu(g0, V, bid);
	s_a[operation_id] = add_mod_ntt_gpu(g0, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 1];
	V = mul_mod_ntt_gpu(g3, S, bid);
	s_a[operation_id + t * 3] = sub_mod_ntt_gpu(g2, V, bid);
	s_a[operation_id + t * 2] = add_mod_ntt_gpu(g2, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 2];
	V = mul_mod_ntt_gpu(g5, S, bid);
	s_a[operation_id + t * 5] = sub_mod_ntt_gpu(g4, V, bid);
	s_a[operation_id + t * 4] = add_mod_ntt_gpu(g4, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 3];
	V = mul_mod_ntt_gpu(g7, S, bid);
	s_a[operation_id + t * 7] = sub_mod_ntt_gpu(g6, V, bid);
	s_a[operation_id + t * 6] = add_mod_ntt_gpu(g6, V, bid);
	__syncthreads();

	a[repeat*gridDim.x*SIFE_N + thread_id + blockIdx.x * SIFE_N] = s_a[thread_id];
	a[repeat*gridDim.x*SIFE_N + thread_id + 512 + blockIdx.x * SIFE_N] = s_a[thread_id + 512];
	a[repeat*gridDim.x*SIFE_N + thread_id + 1024 + blockIdx.x * SIFE_N] = s_a[thread_id + 1024];
	a[repeat*gridDim.x*SIFE_N + thread_id + 1536 + blockIdx.x * SIFE_N] = s_a[thread_id + 1536];
	a[repeat*gridDim.x*SIFE_N + thread_id + 2048 + blockIdx.x * SIFE_N] = s_a[thread_id + 2048];
	a[repeat*gridDim.x*SIFE_N + thread_id + 2560 + blockIdx.x * SIFE_N] = s_a[thread_id + 2560];
	a[repeat*gridDim.x*SIFE_N + thread_id + 3072 + blockIdx.x * SIFE_N] = s_a[thread_id + 3072];
	a[repeat*gridDim.x*SIFE_N + thread_id + 3584 + blockIdx.x * SIFE_N] = s_a[thread_id + 3584];
}

__global__ void GS_reverse_gpu_1block_3round(uint32_t* a) {
	int64_t t, S, U, g0, g1, g2, g3, g4, g5, g6, g7;
	uint32_t thread_id, operation_id;
	__shared__ uint32_t s_a[SIFE_N];
	uint32_t bid = blockIdx.x%SIFE_NMODULI;
	uint32_t repeat = blockIdx.y;

	thread_id = threadIdx.x;
	s_a[thread_id] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + blockIdx.x * SIFE_N];
	s_a[thread_id + 512] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 512 + blockIdx.x * SIFE_N];
	s_a[thread_id + 1024] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 1024 + blockIdx.x * SIFE_N];
	s_a[thread_id + 1536] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 1536 + blockIdx.x * SIFE_N];
	s_a[thread_id + 2048] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 2048 + blockIdx.x * SIFE_N];
	s_a[thread_id + 2560] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 2560 + blockIdx.x * SIFE_N];
	s_a[thread_id + 3072] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 3072 + blockIdx.x * SIFE_N];
	s_a[thread_id + 3584] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 3584 + blockIdx.x * SIFE_N];
	__syncthreads();

	t = 1;
	operation_id = (threadIdx.x / t) * 8 * t + threadIdx.x % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	U = s_a[operation_id];
	g1 = s_a[operation_id + t];
	g0 = add_mod_ntt_gpu(U, g1, bid);
	g1 = sub_mod_ntt_gpu(U, g1, bid);
	g1 = mul_mod_ntt_gpu(g1, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 1];
	U = s_a[operation_id + t * 2];
	g3 = s_a[operation_id + t * 3];
	g2 = add_mod_ntt_gpu(U, g3, bid);
	g3 = sub_mod_ntt_gpu(U, g3, bid);
	g3 = mul_mod_ntt_gpu(g3, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 2];
	U = s_a[operation_id + t * 4];
	g5 = s_a[operation_id + t * 5];
	g4 = add_mod_ntt_gpu(U, g5, bid);
	g5 = sub_mod_ntt_gpu(U, g5, bid);
	g5 = mul_mod_ntt_gpu(g5, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 3];
	U = s_a[operation_id + t * 6];
	g7 = s_a[operation_id + t * 7];
	g6 = add_mod_ntt_gpu(U, g7, bid);
	g7 = sub_mod_ntt_gpu(U, g7, bid);
	g7 = mul_mod_ntt_gpu(g7, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4];
	U = g0;
	g0 = add_mod_ntt_gpu(U, g2, bid);
	g2 = sub_mod_ntt_gpu(U, g2, bid);
	g2 = mul_mod_ntt_gpu(g2, S, bid);
	U = g1;
	g1 = add_mod_ntt_gpu(U, g3, bid);
	g3 = sub_mod_ntt_gpu(U, g3, bid);
	g3 = mul_mod_ntt_gpu(g3, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4 + 1];
	U = g4;
	g4 = add_mod_ntt_gpu(U, g6, bid);
	g6 = sub_mod_ntt_gpu(U, g6, bid);
	g6 = mul_mod_ntt_gpu(g6, S, bid);
	U = g5;
	g5 = add_mod_ntt_gpu(U, g7, bid);
	g7 = sub_mod_ntt_gpu(U, g7, bid);
	g7 = mul_mod_ntt_gpu(g7, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 8 + operation_id / t / 8];
	s_a[operation_id] = add_mod_ntt_gpu(g0, g4, bid);
	g4 = sub_mod_ntt_gpu(g0, g4, bid);
	s_a[operation_id + t * 4] = mul_mod_ntt_gpu(g4, S, bid);
	s_a[operation_id + t] = add_mod_ntt_gpu(g1, g5, bid);
	g5 = sub_mod_ntt_gpu(g1, g5, bid);
	s_a[operation_id + t * 5] = mul_mod_ntt_gpu(g5, S, bid);
	s_a[operation_id + t * 2] = add_mod_ntt_gpu(g2, g6, bid);
	g6 = sub_mod_ntt_gpu(g2, g6, bid);
	s_a[operation_id + t * 6] = mul_mod_ntt_gpu(g6, S, bid);
	s_a[operation_id + t * 3] = add_mod_ntt_gpu(g3, g7, bid);
	g7 = sub_mod_ntt_gpu(g3, g7, bid);
	s_a[operation_id + t * 7] = mul_mod_ntt_gpu(g7, S, bid);
	t = t * 8;
	__syncthreads();

	operation_id = (threadIdx.x / t) * 8 * t + threadIdx.x % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	U = s_a[operation_id];
	g1 = s_a[operation_id + t];
	g0 = add_mod_ntt_gpu(U, g1, bid);
	g1 = sub_mod_ntt_gpu(U, g1, bid);
	g1 = mul_mod_ntt_gpu(g1, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 1];
	U = s_a[operation_id + t * 2];
	g3 = s_a[operation_id + t * 3];
	g2 = add_mod_ntt_gpu(U, g3, bid);
	g3 = sub_mod_ntt_gpu(U, g3, bid);
	g3 = mul_mod_ntt_gpu(g3, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 2];
	U = s_a[operation_id + t * 4];
	g5 = s_a[operation_id + t * 5];
	g4 = add_mod_ntt_gpu(U, g5, bid);
	g5 = sub_mod_ntt_gpu(U, g5, bid);
	g5 = mul_mod_ntt_gpu(g5, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 3];
	U = s_a[operation_id + t * 6];
	g7 = s_a[operation_id + t * 7];
	g6 = add_mod_ntt_gpu(U, g7, bid);
	g7 = sub_mod_ntt_gpu(U, g7, bid);
	g7 = mul_mod_ntt_gpu(g7, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4];
	U = g0;
	g0 = add_mod_ntt_gpu(U, g2, bid);
	g2 = sub_mod_ntt_gpu(U, g2, bid);
	g2 = mul_mod_ntt_gpu(g2, S, bid);
	U = g1;
	g1 = add_mod_ntt_gpu(U, g3, bid);
	g3 = sub_mod_ntt_gpu(U, g3, bid);
	g3 = mul_mod_ntt_gpu(g3, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4 + 1];
	U = g4;
	g4 = add_mod_ntt_gpu(U, g6, bid);
	g6 = sub_mod_ntt_gpu(U, g6, bid);
	g6 = mul_mod_ntt_gpu(g6, S, bid);
	U = g5;
	g5 = add_mod_ntt_gpu(U, g7, bid);
	g7 = sub_mod_ntt_gpu(U, g7, bid);
	g7 = mul_mod_ntt_gpu(g7, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 8 + operation_id / t / 8];
	s_a[operation_id] = add_mod_ntt_gpu(g0, g4, bid);
	g4 = sub_mod_ntt_gpu(g0, g4, bid);
	s_a[operation_id + t * 4] = mul_mod_ntt_gpu(g4, S, bid);
	s_a[operation_id + t] = add_mod_ntt_gpu(g1, g5, bid);
	g5 = sub_mod_ntt_gpu(g1, g5, bid);
	s_a[operation_id + t * 5] = mul_mod_ntt_gpu(g5, S, bid);
	s_a[operation_id + t * 2] = add_mod_ntt_gpu(g2, g6, bid);
	g6 = sub_mod_ntt_gpu(g2, g6, bid);
	s_a[operation_id + t * 6] = mul_mod_ntt_gpu(g6, S, bid);
	s_a[operation_id + t * 3] = add_mod_ntt_gpu(g3, g7, bid);
	g7 = sub_mod_ntt_gpu(g3, g7, bid);
	s_a[operation_id + t * 7] = mul_mod_ntt_gpu(g7, S, bid);
	t = t * 8;
	__syncthreads();

	operation_id = (threadIdx.x / t) * 8 * t + threadIdx.x % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	U = s_a[operation_id];
	g1 = s_a[operation_id + t];
	g0 = add_mod_ntt_gpu(U, g1, bid);
	g1 = sub_mod_ntt_gpu(U, g1, bid);
	g1 = mul_mod_ntt_gpu(g1, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 1];
	U = s_a[operation_id + t * 2];
	g3 = s_a[operation_id + t * 3];
	g2 = add_mod_ntt_gpu(U, g3, bid);
	g3 = sub_mod_ntt_gpu(U, g3, bid);
	g3 = mul_mod_ntt_gpu(g3, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 2];
	U = s_a[operation_id + t * 4];
	g5 = s_a[operation_id + t * 5];
	g4 = add_mod_ntt_gpu(U, g5, bid);
	g5 = sub_mod_ntt_gpu(U, g5, bid);
	g5 = mul_mod_ntt_gpu(g5, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 3];
	U = s_a[operation_id + t * 6];
	g7 = s_a[operation_id + t * 7];
	g6 = add_mod_ntt_gpu(U, g7, bid);
	g7 = sub_mod_ntt_gpu(U, g7, bid);
	g7 = mul_mod_ntt_gpu(g7, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4];
	U = g0;
	g0 = add_mod_ntt_gpu(U, g2, bid);
	g2 = sub_mod_ntt_gpu(U, g2, bid);
	g2 = mul_mod_ntt_gpu(g2, S, bid);
	U = g1;
	g1 = add_mod_ntt_gpu(U, g3, bid);
	g3 = sub_mod_ntt_gpu(U, g3, bid);
	g3 = mul_mod_ntt_gpu(g3, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4 + 1];
	U = g4;
	g4 = add_mod_ntt_gpu(U, g6, bid);
	g6 = sub_mod_ntt_gpu(U, g6, bid);
	g6 = mul_mod_ntt_gpu(g6, S, bid);
	U = g5;
	g5 = add_mod_ntt_gpu(U, g7, bid);
	g7 = sub_mod_ntt_gpu(U, g7, bid);
	g7 = mul_mod_ntt_gpu(g7, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 8 + operation_id / t / 8];
	s_a[operation_id] = add_mod_ntt_gpu(g0, g4, bid);
	g4 = sub_mod_ntt_gpu(g0, g4, bid);
	s_a[operation_id + t * 4] = mul_mod_ntt_gpu(g4, S, bid);
	s_a[operation_id + t] = add_mod_ntt_gpu(g1, g5, bid);
	g5 = sub_mod_ntt_gpu(g1, g5, bid);
	s_a[operation_id + t * 5] = mul_mod_ntt_gpu(g5, S, bid);
	s_a[operation_id + t * 2] = add_mod_ntt_gpu(g2, g6, bid);
	g6 = sub_mod_ntt_gpu(g2, g6, bid);
	s_a[operation_id + t * 6] = mul_mod_ntt_gpu(g6, S, bid);
	s_a[operation_id + t * 3] = add_mod_ntt_gpu(g3, g7, bid);
	g7 = sub_mod_ntt_gpu(g3, g7, bid);
	s_a[operation_id + t * 7] = mul_mod_ntt_gpu(g7, S, bid);
	t = t * 8;
	__syncthreads();

	operation_id = (threadIdx.x / t) * 8 * t + threadIdx.x % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	U = s_a[operation_id];
	g1 = s_a[operation_id + t];
	g0 = add_mod_ntt_gpu(U, g1, bid);
	g1 = sub_mod_ntt_gpu(U, g1, bid);
	g1 = mul_mod_ntt_gpu(g1, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 1];
	U = s_a[operation_id + t * 2];
	g3 = s_a[operation_id + t * 3];
	g2 = add_mod_ntt_gpu(U, g3, bid);
	g3 = sub_mod_ntt_gpu(U, g3, bid);
	g3 = mul_mod_ntt_gpu(g3, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 2];
	U = s_a[operation_id + t * 4];
	g5 = s_a[operation_id + t * 5];
	g4 = add_mod_ntt_gpu(U, g5, bid);
	g5 = sub_mod_ntt_gpu(U, g5, bid);
	g5 = mul_mod_ntt_gpu(g5, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 3];
	U = s_a[operation_id + t * 6];
	g7 = s_a[operation_id + t * 7];
	g6 = add_mod_ntt_gpu(U, g7, bid);
	g7 = sub_mod_ntt_gpu(U, g7, bid);
	g7 = mul_mod_ntt_gpu(g7, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4];
	U = g0;
	g0 = add_mod_ntt_gpu(U, g2, bid);
	g2 = sub_mod_ntt_gpu(U, g2, bid);
	g2 = mul_mod_ntt_gpu(g2, S, bid);
	U = g1;
	g1 = add_mod_ntt_gpu(U, g3, bid);
	g3 = sub_mod_ntt_gpu(U, g3, bid);
	g3 = mul_mod_ntt_gpu(g3, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4 + 1];
	U = g4;
	g4 = add_mod_ntt_gpu(U, g6, bid);
	g6 = sub_mod_ntt_gpu(U, g6, bid);
	g6 = mul_mod_ntt_gpu(g6, S, bid);
	U = g5;
	g5 = add_mod_ntt_gpu(U, g7, bid);
	g7 = sub_mod_ntt_gpu(U, g7, bid);
	g7 = mul_mod_ntt_gpu(g7, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 8 + operation_id / t / 8];
	s_a[operation_id] = add_mod_ntt_gpu(g0, g4, bid);
	g4 = sub_mod_ntt_gpu(g0, g4, bid);
	s_a[operation_id + t * 4] = mul_mod_ntt_gpu(g4, S, bid);
	s_a[operation_id + t] = add_mod_ntt_gpu(g1, g5, bid);
	g5 = sub_mod_ntt_gpu(g1, g5, bid);
	s_a[operation_id + t * 5] = mul_mod_ntt_gpu(g5, S, bid);
	s_a[operation_id + t * 2] = add_mod_ntt_gpu(g2, g6, bid);
	g6 = sub_mod_ntt_gpu(g2, g6, bid);
	s_a[operation_id + t * 6] = mul_mod_ntt_gpu(g6, S, bid);
	s_a[operation_id + t * 3] = add_mod_ntt_gpu(g3, g7, bid);
	g7 = sub_mod_ntt_gpu(g3, g7, bid);
	s_a[operation_id + t * 7] = mul_mod_ntt_gpu(g7, S, bid);
	t = t * 8;
	__syncthreads();

	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + blockIdx.x * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 512 + blockIdx.x * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id + 512], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 1024 + blockIdx.x * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id + 1024], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 1536 + blockIdx.x * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id + 1536], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 2048 + blockIdx.x * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id + 2048], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 2560 + blockIdx.x * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id + 2560], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 3072 + blockIdx.x * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id + 3072], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 3584 + blockIdx.x * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id + 3584], SIFE_NTT_NINV_gpu[bid], bid);
}

__global__ void keygen_gpu(const uint32_t *y, uint32_t *d_msk, uint32_t *d_sky)
{
	uint32_t tid = threadIdx.x, bid = blockIdx.x;
	uint64_t mac=0, i;
	int64_t h1, h2, h3, h4;
	uint32_t w, sel = blockIdx.x % SIFE_NMODULI;
	uint32_t repeat = blockIdx.y;
	__shared__ uint32_t y_crt[SIFE_L];

    //crt_convert_generic_gpu
    if(threadIdx.x < SIFE_L){
    	y_crt[tid] = mod_prime_gpu(y[repeat*SIFE_L + tid], sel);
    }
	__syncthreads();

	//crt_mul_acc_gpu
	h1=0; h2=0; h3=0; h4=0;
	for (i = 0; i < SIFE_L; ++i) {
		w = y_crt[i];
		mac = (uint64_t)w*d_msk[(i*SIFE_NMODULI + bid)*SIFE_N + tid];
		mac = mac + h1;
		h1 = mod_prime_gpu(mac, sel);
		mac = (uint64_t)w*d_msk[(i*SIFE_NMODULI + bid)*SIFE_N + tid + 1024];
		mac = mac + h2;
		h2 = mod_prime_gpu(mac, sel);
		mac = (uint64_t)w*d_msk[(i*SIFE_NMODULI + bid)*SIFE_N + tid + 2048];
		mac = mac + h3;
		h3 = mod_prime_gpu(mac, sel);
		mac = (uint64_t)w*d_msk[(i*SIFE_NMODULI + bid)*SIFE_N + tid + 3072];
		mac = mac + h4;
		h4 = mod_prime_gpu(mac, sel);
	}

	d_sky[(repeat*SIFE_NMODULI + bid)*SIFE_N + tid] = h1;
	d_sky[(repeat*SIFE_NMODULI + bid)*SIFE_N + tid + 1024] = h2;
	d_sky[(repeat*SIFE_NMODULI + bid)*SIFE_N + tid + 2048] = h3;
	d_sky[(repeat*SIFE_NMODULI + bid)*SIFE_N + tid + 3072] = h4;
}

__global__ void decryption_gpu(const uint32_t *y, uint32_t *c, uint32_t* d_sky, uint32_t *dev_dy)
{
	uint32_t tid = threadIdx.x, bid = blockIdx.x;
	uint64_t mac=0, i;
	int64_t t, S, V, U, g1, g2, g3, g4, h1, h2, h3, h4;
	uint32_t w, operation_id, sel = blockIdx.x % SIFE_NMODULI;
	uint32_t repeat = blockIdx.y;
	__shared__ uint32_t y_crt[SIFE_L];
	__shared__ uint32_t s_d1[SIFE_N];
	__shared__ uint32_t s_d2[SIFE_N];

    //crt_convert_generic_gpu
    if(threadIdx.x < SIFE_L){
    	y_crt[tid] = mod_prime_gpu(y[repeat*SIFE_L + tid], sel);
    }

	//CT_forward_gpu_1block_2round
	t = SIFE_N;
	tid = threadIdx.x;
	s_d2[tid] = d_sky[(repeat*SIFE_NMODULI + bid)*SIFE_N + tid];
	s_d2[tid + 1024] = d_sky[(repeat*SIFE_NMODULI + bid)*SIFE_N+ tid + 1024];
	s_d2[tid + 2048] = d_sky[(repeat*SIFE_NMODULI + bid)*SIFE_N+ tid + 2048];
	s_d2[tid + 3072] = d_sky[(repeat*SIFE_NMODULI + bid)*SIFE_N + tid + 3072];
	s_d1[tid] = c[SIFE_L*SIFE_NMODULI*SIFE_N + blockIdx.x*SIFE_N + tid];
	s_d1[tid + 1024] = c[SIFE_L*SIFE_NMODULI*SIFE_N + blockIdx.x*SIFE_N + tid + 1024];
	s_d1[tid + 2048] = c[SIFE_L*SIFE_NMODULI*SIFE_N + blockIdx.x*SIFE_N + tid + 2048];
	s_d1[tid + 3072] = c[SIFE_L*SIFE_NMODULI*SIFE_N + blockIdx.x*SIFE_N + tid + 3072];
	__syncthreads();

	t = t / 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_gpu[sel][SIFE_N / t / 4 + tid / t];
	V = mul_mod_ntt_gpu(s_d2[operation_id + 2 * t], S, sel);
	g1 = add_mod_ntt_gpu(s_d2[operation_id], V, sel);
	g2 = sub_mod_ntt_gpu(s_d2[operation_id], V, sel);
	V = mul_mod_ntt_gpu(s_d2[operation_id + 3 * t], S, sel);
	g3 = add_mod_ntt_gpu(s_d2[operation_id + t], V, sel);
	g4 = sub_mod_ntt_gpu(s_d2[operation_id + t], V, sel);
	V = mul_mod_ntt_gpu(s_d1[operation_id + 2 * t], S, sel);
	h1 = add_mod_ntt_gpu(s_d1[operation_id], V, sel);
	h2 = sub_mod_ntt_gpu(s_d1[operation_id], V, sel);
	V = mul_mod_ntt_gpu(s_d1[operation_id + 3 * t], S, sel);
	h3 = add_mod_ntt_gpu(s_d1[operation_id + t], V, sel);
	h4 = sub_mod_ntt_gpu(s_d1[operation_id + t], V, sel);
	S = psi_gpu[sel][SIFE_N / t / 2 + tid / t * 2];
	V = mul_mod_ntt_gpu(g3, S, sel);
	s_d2[operation_id] = add_mod_ntt_gpu(g1, V, sel);
	s_d2[operation_id + t] = sub_mod_ntt_gpu(g1, V, sel);
	V = mul_mod_ntt_gpu(h3, S, sel);
	s_d1[operation_id] = add_mod_ntt_gpu(h1, V, sel);
	s_d1[operation_id + t] = sub_mod_ntt_gpu(h1, V, sel);
	S = psi_gpu[sel][SIFE_N / t / 2 + tid / t * 2 + 1];
	V = mul_mod_ntt_gpu(g4, S, sel);
	s_d2[operation_id + 2 * t] = add_mod_ntt_gpu(g2, V, sel);
	s_d2[operation_id + 3 * t] = sub_mod_ntt_gpu(g2, V, sel);
	V = mul_mod_ntt_gpu(h4, S, sel);
	s_d1[operation_id + 2 * t] = add_mod_ntt_gpu(h2, V, sel);
	s_d1[operation_id + 3 * t] = sub_mod_ntt_gpu(h2, V, sel);
	__syncthreads();

	t = t / 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_gpu[sel][SIFE_N / t / 4 + tid / t];
	V = mul_mod_ntt_gpu(s_d2[operation_id + 2 * t], S, sel);
	g1 = add_mod_ntt_gpu(s_d2[operation_id], V, sel);
	g2 = sub_mod_ntt_gpu(s_d2[operation_id], V, sel);
	V = mul_mod_ntt_gpu(s_d2[operation_id + 3 * t], S, sel);
	g3 = add_mod_ntt_gpu(s_d2[operation_id + t], V, sel);
	g4 = sub_mod_ntt_gpu(s_d2[operation_id + t], V, sel);
	V = mul_mod_ntt_gpu(s_d1[operation_id + 2 * t], S, sel);
	h1 = add_mod_ntt_gpu(s_d1[operation_id], V, sel);
	h2 = sub_mod_ntt_gpu(s_d1[operation_id], V, sel);
	V = mul_mod_ntt_gpu(s_d1[operation_id + 3 * t], S, sel);
	h3 = add_mod_ntt_gpu(s_d1[operation_id + t], V, sel);
	h4 = sub_mod_ntt_gpu(s_d1[operation_id + t], V, sel);
	S = psi_gpu[sel][SIFE_N / t / 2 + tid / t * 2];
	V = mul_mod_ntt_gpu(g3, S, sel);
	s_d2[operation_id] = add_mod_ntt_gpu(g1, V, sel);
	s_d2[operation_id + t] = sub_mod_ntt_gpu(g1, V, sel);
	V = mul_mod_ntt_gpu(h3, S, sel);
	s_d1[operation_id] = add_mod_ntt_gpu(h1, V, sel);
	s_d1[operation_id + t] = sub_mod_ntt_gpu(h1, V, sel);
	S = psi_gpu[sel][SIFE_N / t / 2 + tid / t * 2 + 1];
	V = mul_mod_ntt_gpu(g4, S, sel);
	s_d2[operation_id + 2 * t] = add_mod_ntt_gpu(g2, V, sel);
	s_d2[operation_id + 3 * t] = sub_mod_ntt_gpu(g2, V, sel);
	V = mul_mod_ntt_gpu(h4, S, sel);
	s_d1[operation_id + 2 * t] = add_mod_ntt_gpu(h2, V, sel);
	s_d1[operation_id + 3 * t] = sub_mod_ntt_gpu(h2, V, sel);
	__syncthreads();

	t = t / 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_gpu[sel][SIFE_N / t / 4 + tid / t];
	V = mul_mod_ntt_gpu(s_d2[operation_id + 2 * t], S, sel);
	g1 = add_mod_ntt_gpu(s_d2[operation_id], V, sel);
	g2 = sub_mod_ntt_gpu(s_d2[operation_id], V, sel);
	V = mul_mod_ntt_gpu(s_d2[operation_id + 3 * t], S, sel);
	g3 = add_mod_ntt_gpu(s_d2[operation_id + t], V, sel);
	g4 = sub_mod_ntt_gpu(s_d2[operation_id + t], V, sel);
	V = mul_mod_ntt_gpu(s_d1[operation_id + 2 * t], S, sel);
	h1 = add_mod_ntt_gpu(s_d1[operation_id], V, sel);
	h2 = sub_mod_ntt_gpu(s_d1[operation_id], V, sel);
	V = mul_mod_ntt_gpu(s_d1[operation_id + 3 * t], S, sel);
	h3 = add_mod_ntt_gpu(s_d1[operation_id + t], V, sel);
	h4 = sub_mod_ntt_gpu(s_d1[operation_id + t], V, sel);
	S = psi_gpu[sel][SIFE_N / t / 2 + tid / t * 2];
	V = mul_mod_ntt_gpu(g3, S, sel);
	s_d2[operation_id] = add_mod_ntt_gpu(g1, V, sel);
	s_d2[operation_id + t] = sub_mod_ntt_gpu(g1, V, sel);
	V = mul_mod_ntt_gpu(h3, S, sel);
	s_d1[operation_id] = add_mod_ntt_gpu(h1, V, sel);
	s_d1[operation_id + t] = sub_mod_ntt_gpu(h1, V, sel);
	S = psi_gpu[sel][SIFE_N / t / 2 + tid / t * 2 + 1];
	V = mul_mod_ntt_gpu(g4, S, sel);
	s_d2[operation_id + 2 * t] = add_mod_ntt_gpu(g2, V, sel);
	s_d2[operation_id + 3 * t] = sub_mod_ntt_gpu(g2, V, sel);
	V = mul_mod_ntt_gpu(h4, S, sel);
	s_d1[operation_id + 2 * t] = add_mod_ntt_gpu(h2, V, sel);
	s_d1[operation_id + 3 * t] = sub_mod_ntt_gpu(h2, V, sel);
	__syncthreads();

	t = t / 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_gpu[sel][SIFE_N / t / 4 + tid / t];
	V = mul_mod_ntt_gpu(s_d2[operation_id + 2 * t], S, sel);
	g1 = add_mod_ntt_gpu(s_d2[operation_id], V, sel);
	g2 = sub_mod_ntt_gpu(s_d2[operation_id], V, sel);
	V = mul_mod_ntt_gpu(s_d2[operation_id + 3 * t], S, sel);
	g3 = add_mod_ntt_gpu(s_d2[operation_id + t], V, sel);
	g4 = sub_mod_ntt_gpu(s_d2[operation_id + t], V, sel);
	V = mul_mod_ntt_gpu(s_d1[operation_id + 2 * t], S, sel);
	h1 = add_mod_ntt_gpu(s_d1[operation_id], V, sel);
	h2 = sub_mod_ntt_gpu(s_d1[operation_id], V, sel);
	V = mul_mod_ntt_gpu(s_d1[operation_id + 3 * t], S, sel);
	h3 = add_mod_ntt_gpu(s_d1[operation_id + t], V, sel);
	h4 = sub_mod_ntt_gpu(s_d1[operation_id + t], V, sel);
	S = psi_gpu[sel][SIFE_N / t / 2 + tid / t * 2];
	V = mul_mod_ntt_gpu(g3, S, sel);
	s_d2[operation_id] = add_mod_ntt_gpu(g1, V, sel);
	s_d2[operation_id + t] = sub_mod_ntt_gpu(g1, V, sel);
	V = mul_mod_ntt_gpu(h3, S, sel);
	s_d1[operation_id] = add_mod_ntt_gpu(h1, V, sel);
	s_d1[operation_id + t] = sub_mod_ntt_gpu(h1, V, sel);
	S = psi_gpu[sel][SIFE_N / t / 2 + tid / t * 2 + 1];
	V = mul_mod_ntt_gpu(g4, S, sel);
	s_d2[operation_id + 2 * t] = add_mod_ntt_gpu(g2, V, sel);
	s_d2[operation_id + 3 * t] = sub_mod_ntt_gpu(g2, V, sel);
	V = mul_mod_ntt_gpu(h4, S, sel);
	s_d1[operation_id + 2 * t] = add_mod_ntt_gpu(h2, V, sel);
	s_d1[operation_id + 3 * t] = sub_mod_ntt_gpu(h2, V, sel);
	__syncthreads();

	t = t / 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_gpu[sel][SIFE_N / t / 4 + tid / t];
	V = mul_mod_ntt_gpu(s_d2[operation_id + 2 * t], S, sel);
	g1 = add_mod_ntt_gpu(s_d2[operation_id], V, sel);
	g2 = sub_mod_ntt_gpu(s_d2[operation_id], V, sel);
	V = mul_mod_ntt_gpu(s_d2[operation_id + 3 * t], S, sel);
	g3 = add_mod_ntt_gpu(s_d2[operation_id + t], V, sel);
	g4 = sub_mod_ntt_gpu(s_d2[operation_id + t], V, sel);
	V = mul_mod_ntt_gpu(s_d1[operation_id + 2 * t], S, sel);
	h1 = add_mod_ntt_gpu(s_d1[operation_id], V, sel);
	h2 = sub_mod_ntt_gpu(s_d1[operation_id], V, sel);
	V = mul_mod_ntt_gpu(s_d1[operation_id + 3 * t], S, sel);
	h3 = add_mod_ntt_gpu(s_d1[operation_id + t], V, sel);
	h4 = sub_mod_ntt_gpu(s_d1[operation_id + t], V, sel);
	S = psi_gpu[sel][SIFE_N / t / 2 + tid / t * 2];
	V = mul_mod_ntt_gpu(g3, S, sel);
	s_d2[operation_id] = add_mod_ntt_gpu(g1, V, sel);
	s_d2[operation_id + t] = sub_mod_ntt_gpu(g1, V, sel);
	V = mul_mod_ntt_gpu(h3, S, sel);
	s_d1[operation_id] = add_mod_ntt_gpu(h1, V, sel);
	s_d1[operation_id + t] = sub_mod_ntt_gpu(h1, V, sel);
	S = psi_gpu[sel][SIFE_N / t / 2 + tid / t * 2 + 1];
	V = mul_mod_ntt_gpu(g4, S, sel);
	s_d2[operation_id + 2 * t] = add_mod_ntt_gpu(g2, V, sel);
	s_d2[operation_id + 3 * t] = sub_mod_ntt_gpu(g2, V, sel);
	V = mul_mod_ntt_gpu(h4, S, sel);
	s_d1[operation_id + 2 * t] = add_mod_ntt_gpu(h2, V, sel);
	s_d1[operation_id + 3 * t] = sub_mod_ntt_gpu(h2, V, sel);
	__syncthreads();

	t = t / 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_gpu[sel][SIFE_N / t / 4 + tid / t];
	V = mul_mod_ntt_gpu(s_d2[operation_id + 2 * t], S, sel);
	g1 = add_mod_ntt_gpu(s_d2[operation_id], V, sel);
	g2 = sub_mod_ntt_gpu(s_d2[operation_id], V, sel);
	V = mul_mod_ntt_gpu(s_d2[operation_id + 3 * t], S, sel);
	g3 = add_mod_ntt_gpu(s_d2[operation_id + t], V, sel);
	g4 = sub_mod_ntt_gpu(s_d2[operation_id + t], V, sel);
	V = mul_mod_ntt_gpu(s_d1[operation_id + 2 * t], S, sel);
	h1 = add_mod_ntt_gpu(s_d1[operation_id], V, sel);
	h2 = sub_mod_ntt_gpu(s_d1[operation_id], V, sel);
	V = mul_mod_ntt_gpu(s_d1[operation_id + 3 * t], S, sel);
	h3 = add_mod_ntt_gpu(s_d1[operation_id + t], V, sel);
	h4 = sub_mod_ntt_gpu(s_d1[operation_id + t], V, sel);
	S = psi_gpu[sel][SIFE_N / t / 2 + tid / t * 2];
	V = mul_mod_ntt_gpu(g3, S, sel);
	s_d2[operation_id] = add_mod_ntt_gpu(g1, V, sel);
	s_d2[operation_id + t] = sub_mod_ntt_gpu(g1, V, sel);
	V = mul_mod_ntt_gpu(h3, S, sel);
	s_d1[operation_id] = add_mod_ntt_gpu(h1, V, sel);
	s_d1[operation_id + t] = sub_mod_ntt_gpu(h1, V, sel);
	S = psi_gpu[sel][SIFE_N / t / 2 + tid / t * 2 + 1];
	V = mul_mod_ntt_gpu(g4, S, sel);
	s_d2[operation_id + 2 * t] = add_mod_ntt_gpu(g2, V, sel);
	s_d2[operation_id + 3 * t] = sub_mod_ntt_gpu(g2, V, sel);
	V = mul_mod_ntt_gpu(h4, S, sel);
	s_d1[operation_id + 2 * t] = add_mod_ntt_gpu(h2, V, sel);
	s_d1[operation_id + 3 * t] = sub_mod_ntt_gpu(h2, V, sel);
	__syncthreads();

	//point_mul_gpu3
	s_d2[tid] = mul_mod_ntt_gpu(s_d2[tid], s_d1[tid], sel);
	s_d2[tid + 1024] = mul_mod_ntt_gpu(s_d2[tid + 1024], s_d1[tid + 1024], sel);
	s_d2[tid + 2048] = mul_mod_ntt_gpu(s_d2[tid + 2048], s_d1[tid + 2048], sel);
	s_d2[tid + 3072] = mul_mod_ntt_gpu(s_d2[tid + 3072], s_d1[tid + 3072], sel);	
	__syncthreads();

	//GS_reverse_gpu_1block_2round
	t = 1;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (tid / t) * 2];
	U = s_d2[operation_id];
	g2 = s_d2[operation_id + t];
	g1 = add_mod_ntt_gpu(U, g2, sel);
	g2 = sub_mod_ntt_gpu(U, g2, sel);
	g2 = mul_mod_ntt_gpu(g2, S, sel);
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (tid / t) * 2 + 1];
	U = s_d2[operation_id + 2 * t];
	g4 = s_d2[operation_id + 3 * t];
	g3 = add_mod_ntt_gpu(U, g4, sel);
	g4 = sub_mod_ntt_gpu(U, g4, sel);
	g4 = mul_mod_ntt_gpu(g4, S, sel);
	S = psi_inv_gpu[sel][SIFE_N / t / 4 + tid / t];
	U = g1;
	s_d2[operation_id] = add_mod_ntt_gpu(U, g3, sel);
	g3 = sub_mod_ntt_gpu(U, g3, sel);
	s_d2[operation_id + 2 * t] = mul_mod_ntt_gpu(g3, S, sel);
	U = g2;
	s_d2[operation_id + t] = add_mod_ntt_gpu(U, g4, sel);
	g4 = sub_mod_ntt_gpu(U, g4, sel);
	s_d2[operation_id + 3 * t] = mul_mod_ntt_gpu(g4, S, sel);
	__syncthreads();

	t = t * 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (tid / t) * 2];
	U = s_d2[operation_id];
	g2 = s_d2[operation_id + t];
	g1 = add_mod_ntt_gpu(U, g2, sel);
	g2 = sub_mod_ntt_gpu(U, g2, sel);
	g2 = mul_mod_ntt_gpu(g2, S, sel);
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (tid / t) * 2 + 1];
	U = s_d2[operation_id + 2 * t];
	g4 = s_d2[operation_id + 3 * t];
	g3 = add_mod_ntt_gpu(U, g4, sel);
	g4 = sub_mod_ntt_gpu(U, g4, sel);
	g4 = mul_mod_ntt_gpu(g4, S, sel);
	S = psi_inv_gpu[sel][SIFE_N / t / 4 + tid / t];
	U = g1;
	s_d2[operation_id] = add_mod_ntt_gpu(U, g3, sel);
	g3 = sub_mod_ntt_gpu(U, g3, sel);
	s_d2[operation_id + 2 * t] = mul_mod_ntt_gpu(g3, S, sel);
	U = g2;
	s_d2[operation_id + t] = add_mod_ntt_gpu(U, g4, sel);
	g4 = sub_mod_ntt_gpu(U, g4, sel);
	s_d2[operation_id + 3 * t] = mul_mod_ntt_gpu(g4, S, sel);
	__syncthreads();

	t = t * 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (tid / t) * 2];
	U = s_d2[operation_id];
	g2 = s_d2[operation_id + t];
	g1 = add_mod_ntt_gpu(U, g2, sel);
	g2 = sub_mod_ntt_gpu(U, g2, sel);
	g2 = mul_mod_ntt_gpu(g2, S, sel);
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (tid / t) * 2 + 1];
	U = s_d2[operation_id + 2 * t];
	g4 = s_d2[operation_id + 3 * t];
	g3 = add_mod_ntt_gpu(U, g4, sel);
	g4 = sub_mod_ntt_gpu(U, g4, sel);
	g4 = mul_mod_ntt_gpu(g4, S, sel);
	S = psi_inv_gpu[sel][SIFE_N / t / 4 + tid / t];
	U = g1;
	s_d2[operation_id] = add_mod_ntt_gpu(U, g3, sel);
	g3 = sub_mod_ntt_gpu(U, g3, sel);
	s_d2[operation_id + 2 * t] = mul_mod_ntt_gpu(g3, S, sel);
	U = g2;
	s_d2[operation_id + t] = add_mod_ntt_gpu(U, g4, sel);
	g4 = sub_mod_ntt_gpu(U, g4, sel);
	s_d2[operation_id + 3 * t] = mul_mod_ntt_gpu(g4, S, sel);
	__syncthreads();

	t = t * 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (tid / t) * 2];
	U = s_d2[operation_id];
	g2 = s_d2[operation_id + t];
	g1 = add_mod_ntt_gpu(U, g2, sel);
	g2 = sub_mod_ntt_gpu(U, g2, sel);
	g2 = mul_mod_ntt_gpu(g2, S, sel);
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (tid / t) * 2 + 1];
	U = s_d2[operation_id + 2 * t];
	g4 = s_d2[operation_id + 3 * t];
	g3 = add_mod_ntt_gpu(U, g4, sel);
	g4 = sub_mod_ntt_gpu(U, g4, sel);
	g4 = mul_mod_ntt_gpu(g4, S, sel);
	S = psi_inv_gpu[sel][SIFE_N / t / 4 + tid / t];
	U = g1;
	s_d2[operation_id] = add_mod_ntt_gpu(U, g3, sel);
	g3 = sub_mod_ntt_gpu(U, g3, sel);
	s_d2[operation_id + 2 * t] = mul_mod_ntt_gpu(g3, S, sel);
	U = g2;
	s_d2[operation_id + t] = add_mod_ntt_gpu(U, g4, sel);
	g4 = sub_mod_ntt_gpu(U, g4, sel);
	s_d2[operation_id + 3 * t] = mul_mod_ntt_gpu(g4, S, sel);
	__syncthreads();

	t = t * 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (tid / t) * 2];
	U = s_d2[operation_id];
	g2 = s_d2[operation_id + t];
	g1 = add_mod_ntt_gpu(U, g2, sel);
	g2 = sub_mod_ntt_gpu(U, g2, sel);
	g2 = mul_mod_ntt_gpu(g2, S, sel);
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (tid / t) * 2 + 1];
	U = s_d2[operation_id + 2 * t];
	g4 = s_d2[operation_id + 3 * t];
	g3 = add_mod_ntt_gpu(U, g4, sel);
	g4 = sub_mod_ntt_gpu(U, g4, sel);
	g4 = mul_mod_ntt_gpu(g4, S, sel);
	S = psi_inv_gpu[sel][SIFE_N / t / 4 + tid / t];
	U = g1;
	s_d2[operation_id] = add_mod_ntt_gpu(U, g3, sel);
	g3 = sub_mod_ntt_gpu(U, g3, sel);
	s_d2[operation_id + 2 * t] = mul_mod_ntt_gpu(g3, S, sel);
	U = g2;
	s_d2[operation_id + t] = add_mod_ntt_gpu(U, g4, sel);
	g4 = sub_mod_ntt_gpu(U, g4, sel);
	s_d2[operation_id + 3 * t] = mul_mod_ntt_gpu(g4, S, sel);
	__syncthreads();

	t = t * 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (tid / t) * 2];
	U = s_d2[operation_id];
	g2 = s_d2[operation_id + t];
	g1 = add_mod_ntt_gpu(U, g2, sel);
	g2 = sub_mod_ntt_gpu(U, g2, sel);
	g2 = mul_mod_ntt_gpu(g2, S, sel);
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (tid / t) * 2 + 1];
	U = s_d2[operation_id + 2 * t];
	g4 = s_d2[operation_id + 3 * t];
	g3 = add_mod_ntt_gpu(U, g4, sel);
	g4 = sub_mod_ntt_gpu(U, g4, sel);
	g4 = mul_mod_ntt_gpu(g4, S, sel);
	S = psi_inv_gpu[sel][SIFE_N / t / 4 + tid / t];
	U = g1;
	s_d2[operation_id] = add_mod_ntt_gpu(U, g3, sel);
	g3 = sub_mod_ntt_gpu(U, g3, sel);
	s_d2[operation_id + 2 * t] = mul_mod_ntt_gpu(g3, S, sel);
	U = g2;
	s_d2[operation_id + t] = add_mod_ntt_gpu(U, g4, sel);
	g4 = sub_mod_ntt_gpu(U, g4, sel);
	s_d2[operation_id + 3 * t] = mul_mod_ntt_gpu(g4, S, sel);
	__syncthreads();

	g1 = mul_mod_ntt_gpu(s_d2[tid], SIFE_NTT_NINV_gpu[sel], sel);
	g2 = mul_mod_ntt_gpu(s_d2[tid + 1024], SIFE_NTT_NINV_gpu[sel], sel);
	g3 = mul_mod_ntt_gpu(s_d2[tid + 2048], SIFE_NTT_NINV_gpu[sel], sel);
	g4 = mul_mod_ntt_gpu(s_d2[tid + 3072], SIFE_NTT_NINV_gpu[sel], sel);

	//crt_mul_acc_gpu
	h1=0; h2=0; h3=0; h4=0;
	for (i = 0; i < SIFE_L; ++i) {
		w = y_crt[i];
		mac = (uint64_t)w*c[(i*SIFE_NMODULI + bid)*SIFE_N + tid];
		mac = mac + h1;
		h1 = mod_prime_gpu(mac, sel);
		mac = (uint64_t)w*c[(i*SIFE_NMODULI + bid)*SIFE_N + tid + 1024];
		mac = mac + h2;
		h2 = mod_prime_gpu(mac, sel);
		mac = (uint64_t)w*c[(i*SIFE_NMODULI + bid)*SIFE_N + tid + 2048];
		mac = mac + h3;
		h3 = mod_prime_gpu(mac, sel);
		mac = (uint64_t)w*c[(i*SIFE_NMODULI + bid)*SIFE_N + tid + 3072];
		mac = mac + h4;
		h4 = mod_prime_gpu(mac, sel);
	}

	//poly_sub_mod_gpu
	dev_dy[(repeat*SIFE_NMODULI + bid)*SIFE_N + tid] = sub_mod_ntt_gpu(h1, g1, sel);
	dev_dy[(repeat*SIFE_NMODULI + bid)*SIFE_N + tid + 1024] = sub_mod_ntt_gpu(h2, g2, sel);
	dev_dy[(repeat*SIFE_NMODULI + bid)*SIFE_N + tid + 2048] = sub_mod_ntt_gpu(h3, g3, sel);
	dev_dy[(repeat*SIFE_NMODULI + bid)*SIFE_N + tid + 3072] = sub_mod_ntt_gpu(h4, g4, sel);
}