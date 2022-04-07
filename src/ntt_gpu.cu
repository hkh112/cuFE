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
	uint32_t tid = threadIdx.x, bid = blockIdx.x*SIFE_N;
	int32_t i;
	for (i = 0; i < SIFE_N/1024; ++i)
	{
		d_c[bid+tid+i*1024] = mul_mod_ntt_gpu(d_a[bid + tid+i*1024], d_b[bid+tid+i*1024], blockIdx.x);			
	}	
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
	int32_t j, i;
	for (j = 0; j < SIFE_NMODULI; ++j)
	{
		for (i = 0; i < SIFE_N/1024; ++i)
		{
			d_c[bid+tid+j*SIFE_N+i*+1024] = mod_prime_gpu(d_c[bid+tid+j*SIFE_N+i*+1024] + d_m[bid2 + j*SIFE_L], j);
		}
	}	
}

__global__ void point_add_mod_gpu3(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b)
{
	uint32_t tid = threadIdx.x, bid = blockIdx.x*SIFE_N;
	int32_t i;
	for (i = 0; i < SIFE_N/1024; ++i)
	{
		d_c[bid+tid+i*1024] = add_mod_ntt_gpu(d_a[bid+tid+i*1024], d_b[bid+tid+i*1024], blockIdx.x);
	}	
}

__global__ void poly_sub_mod_gpu(uint32_t *a, uint32_t *b, uint32_t *c)
{
	uint32_t tid = threadIdx.x, bid = blockIdx.x*SIFE_N, i;
	for (i = 0; i < SIFE_N/1024; ++i)
	{
		c[bid + tid+i*1024] = sub_mod_ntt_gpu(a[bid + tid+i*1024], b[bid + tid+i*1024], blockIdx.x);
	// c[bid + tid+1024] = sub_mod_ntt_gpu(a[bid + tid+1024], b[bid + tid+1024], blockIdx.x);
	}
}

__global__ void CT_forward_gpu_1block_1round(uint32_t a[SIFE_N]) {
	int64_t t, S, V;
	uint32_t tid = threadIdx.x, bid = blockIdx.x%SIFE_NMODULI;
	uint32_t operation_id;
	__shared__ uint32_t s_a[SIFE_N];

	t = SIFE_N;
	s_a[tid] = a[blockIdx.x*SIFE_N + tid];
	s_a[tid + 1024] = a[blockIdx.x*SIFE_N + tid + 1024];
	__syncthreads();

	t = t / 2;
	operation_id = tid;
	S = psi_gpu[bid][SIFE_N / t / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + t], S, bid);
	s_a[operation_id + t] = sub_mod_ntt_gpu(s_a[operation_id], V, bid);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, bid);
	__syncthreads();

	t = t / 2;
	operation_id = (tid / t) * 2 * t + tid % t;
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + t], S, bid);
	s_a[operation_id + t] = sub_mod_ntt_gpu(s_a[operation_id], V, bid);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, bid);
	__syncthreads();

	t = t / 2;
	operation_id = (tid / t) * 2 * t + tid % t;
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + t], S, bid);
	s_a[operation_id + t] = sub_mod_ntt_gpu(s_a[operation_id], V, bid);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, bid);
	__syncthreads();

	t = t / 2;
	operation_id = (tid / t) * 2 * t + tid % t;
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + t], S, bid);
	s_a[operation_id + t] = sub_mod_ntt_gpu(s_a[operation_id], V, bid);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, bid);
	__syncthreads();

	t = t / 2;
	operation_id = (tid / t) * 2 * t + tid % t;
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + t], S, bid);
	s_a[operation_id + t] = sub_mod_ntt_gpu(s_a[operation_id], V, bid);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, bid);
	__syncthreads();

	t = t / 2;
	operation_id = (tid / t) * 2 * t + tid % t;
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + t], S, bid);
	s_a[operation_id + t] = sub_mod_ntt_gpu(s_a[operation_id], V, bid);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, bid);
	__syncthreads();

	t = t / 2;
	operation_id = (tid / t) * 2 * t + tid % t;
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + t], S, bid);
	s_a[operation_id + t] = sub_mod_ntt_gpu(s_a[operation_id], V, bid);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, bid);
	__syncthreads();

	t = t / 2;
	operation_id = (tid / t) * 2 * t + tid % t;
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + t], S, bid);
	s_a[operation_id + t] = sub_mod_ntt_gpu(s_a[operation_id], V, bid);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, bid);
	__syncthreads();

	t = t / 2;
	operation_id = (tid / t) * 2 * t + tid % t;
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + t], S, bid);
	s_a[operation_id + t] = sub_mod_ntt_gpu(s_a[operation_id], V, bid);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, bid);
	__syncthreads();

	t = t / 2;
	operation_id = (tid / t) * 2 * t + tid % t;
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + t], S, bid);
	s_a[operation_id + t] = sub_mod_ntt_gpu(s_a[operation_id], V, bid);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, bid);
	__syncthreads();

	t = t / 2;
	operation_id = (tid / t) * 2 * t + tid % t;
	S = psi_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + t], S, bid);
	s_a[operation_id + t] = sub_mod_ntt_gpu(s_a[operation_id], V, bid);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, bid);
	__syncthreads();

	a[blockIdx.x*SIFE_N + tid] = s_a[tid];
	a[blockIdx.x*SIFE_N + tid + 1024] = s_a[tid + 1024];
}

__global__ void GS_reverse_gpu_1block_1round(uint32_t a[SIFE_N]) {
	int64_t t, S, U, V;
	uint32_t operation_id;
	__shared__ uint32_t s_a[SIFE_N];
	uint32_t tid = threadIdx.x, bid = blockIdx.x%SIFE_NMODULI;

	t = 1;
	s_a[tid] = a[blockIdx.x*SIFE_N + tid];
	s_a[tid + 1024] = a[blockIdx.x*SIFE_N + tid + 1024];
	__syncthreads();

	operation_id = (tid / t) * 2 * t + tid % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + t];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, bid);
	V = sub_mod_ntt_gpu(U, V, bid);
	s_a[operation_id + t] = mul_mod_ntt_gpu(V, S, bid);
	t = t * 2;
	__syncthreads();

	operation_id = (tid / t) * 2 * t + tid % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + t];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, bid);
	V = sub_mod_ntt_gpu(U, V, bid);
	s_a[operation_id + t] = mul_mod_ntt_gpu(V, S, bid);
	t = t * 2;
	__syncthreads();

	operation_id = (tid / t) * 2 * t + tid % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + t];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, bid);
	V = sub_mod_ntt_gpu(U, V, bid);
	s_a[operation_id + t] = mul_mod_ntt_gpu(V, S, bid);
	t = t * 2;
	__syncthreads();

	operation_id = (tid / t) * 2 * t + tid % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + t];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, bid);
	V = sub_mod_ntt_gpu(U, V, bid);
	s_a[operation_id + t] = mul_mod_ntt_gpu(V, S, bid);
	t = t * 2;
	__syncthreads();

	operation_id = (tid / t) * 2 * t + tid % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + t];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, bid);
	V = sub_mod_ntt_gpu(U, V, bid);
	s_a[operation_id + t] = mul_mod_ntt_gpu(V, S, bid);
	t = t * 2;
	__syncthreads();

	operation_id = (tid / t) * 2 * t + tid % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + t];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, bid);
	V = sub_mod_ntt_gpu(U, V, bid);
	s_a[operation_id + t] = mul_mod_ntt_gpu(V, S, bid);
	t = t * 2;
	__syncthreads();

	operation_id = (tid / t) * 2 * t + tid % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + t];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, bid);
	V = sub_mod_ntt_gpu(U, V, bid);
	s_a[operation_id + t] = mul_mod_ntt_gpu(V, S, bid);
	t = t * 2;
	__syncthreads();

	operation_id = (tid / t) * 2 * t + tid % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + t];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, bid);
	V = sub_mod_ntt_gpu(U, V, bid);
	s_a[operation_id + t] = mul_mod_ntt_gpu(V, S, bid);
	t = t * 2;
	__syncthreads();

	operation_id = (tid / t) * 2 * t + tid % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + t];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, bid);
	V = sub_mod_ntt_gpu(U, V, bid);
	s_a[operation_id + t] = mul_mod_ntt_gpu(V, S, bid);
	t = t * 2;
	__syncthreads();

	operation_id = (tid / t) * 2 * t + tid % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + t];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, bid);
	V = sub_mod_ntt_gpu(U, V, bid);
	s_a[operation_id + t] = mul_mod_ntt_gpu(V, S, bid);
	t = t * 2;
	__syncthreads();

	operation_id = tid;
	S = psi_inv_gpu[bid][SIFE_N / t / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + t];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, bid);
	V = sub_mod_ntt_gpu(U, V, bid);
	s_a[operation_id + t] = mul_mod_ntt_gpu(V, S, bid);
	__syncthreads();

	a[blockIdx.x*SIFE_N + tid] = mul_mod_ntt_gpu(s_a[tid], SIFE_NTT_NINV_gpu[bid], bid);
	a[blockIdx.x*SIFE_N + tid + 1024] = mul_mod_ntt_gpu(s_a[tid + 1024], SIFE_NTT_NINV_gpu[bid], bid);
}

__global__ void CT_forward_gpu_1block_2round(uint32_t a[SIFE_N]) {
	int64_t t, S, V, g1, g2, g3, g4;
	uint32_t tid, operation_id, bid = blockIdx.x%SIFE_NMODULI;
	uint32_t repeat = blockIdx.y;
	__shared__ uint32_t s_a[SIFE_N];

	t = SIFE_N;
	tid = threadIdx.x;
	s_a[tid] = a[repeat*gridDim.x*SIFE_N + blockIdx.x*SIFE_N + tid];
	s_a[tid + 1024] = a[repeat*gridDim.x*SIFE_N + blockIdx.x*SIFE_N + tid + 1024];
	s_a[tid + 2048] = a[repeat*gridDim.x*SIFE_N + blockIdx.x*SIFE_N + tid + 2048];
	s_a[tid + 3072] = a[repeat*gridDim.x*SIFE_N + blockIdx.x*SIFE_N + tid + 3072];

	t = t / 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_gpu[bid][SIFE_N / t / 4 + tid / t];
	V = mul_mod_ntt_gpu(s_a[operation_id + 2 * t], S, bid);
	g1 = add_mod_ntt_gpu(s_a[operation_id], V, bid);
	g2 = sub_mod_ntt_gpu(s_a[operation_id], V, bid);
	V = mul_mod_ntt_gpu(s_a[operation_id + 3 * t], S, bid);
	g3 = add_mod_ntt_gpu(s_a[operation_id + t], V, bid);
	g4 = sub_mod_ntt_gpu(s_a[operation_id + t], V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + tid / t * 2];
	V = mul_mod_ntt_gpu(g3, S, bid);
	s_a[operation_id] = add_mod_ntt_gpu(g1, V, bid);
	s_a[operation_id + t] = sub_mod_ntt_gpu(g1, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + tid / t * 2 + 1];
	V = mul_mod_ntt_gpu(g4, S, bid);
	s_a[operation_id + 2 * t] = add_mod_ntt_gpu(g2, V, bid);
	s_a[operation_id + 3 * t] = sub_mod_ntt_gpu(g2, V, bid);
	__syncthreads();

	t = t / 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_gpu[bid][SIFE_N / t / 4 + tid / t];
	V = mul_mod_ntt_gpu(s_a[operation_id + 2 * t], S, bid);
	g1 = add_mod_ntt_gpu(s_a[operation_id], V, bid);
	g2 = sub_mod_ntt_gpu(s_a[operation_id], V, bid);
	V = mul_mod_ntt_gpu(s_a[operation_id + 3 * t], S, bid);
	g3 = add_mod_ntt_gpu(s_a[operation_id + t], V, bid);
	g4 = sub_mod_ntt_gpu(s_a[operation_id + t], V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + tid / t * 2];
	V = mul_mod_ntt_gpu(g3, S, bid);
	s_a[operation_id] = add_mod_ntt_gpu(g1, V, bid);
	s_a[operation_id + t] = sub_mod_ntt_gpu(g1, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + tid / t * 2 + 1];
	V = mul_mod_ntt_gpu(g4, S, bid);
	s_a[operation_id + 2 * t] = add_mod_ntt_gpu(g2, V, bid);
	s_a[operation_id + 3 * t] = sub_mod_ntt_gpu(g2, V, bid);
	__syncthreads();

	t = t / 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_gpu[bid][SIFE_N / t / 4 + tid / t];
	V = mul_mod_ntt_gpu(s_a[operation_id + 2 * t], S, bid);
	g1 = add_mod_ntt_gpu(s_a[operation_id], V, bid);
	g2 = sub_mod_ntt_gpu(s_a[operation_id], V, bid);
	V = mul_mod_ntt_gpu(s_a[operation_id + 3 * t], S, bid);
	g3 = add_mod_ntt_gpu(s_a[operation_id + t], V, bid);
	g4 = sub_mod_ntt_gpu(s_a[operation_id + t], V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + tid / t * 2];
	V = mul_mod_ntt_gpu(g3, S, bid);
	s_a[operation_id] = add_mod_ntt_gpu(g1, V, bid);
	s_a[operation_id + t] = sub_mod_ntt_gpu(g1, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + tid / t * 2 + 1];
	V = mul_mod_ntt_gpu(g4, S, bid);
	s_a[operation_id + 2 * t] = add_mod_ntt_gpu(g2, V, bid);
	s_a[operation_id + 3 * t] = sub_mod_ntt_gpu(g2, V, bid);
	__syncthreads();

	t = t / 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_gpu[bid][SIFE_N / t / 4 + tid / t];
	V = mul_mod_ntt_gpu(s_a[operation_id + 2 * t], S, bid);
	g1 = add_mod_ntt_gpu(s_a[operation_id], V, bid);
	g2 = sub_mod_ntt_gpu(s_a[operation_id], V, bid);
	V = mul_mod_ntt_gpu(s_a[operation_id + 3 * t], S, bid);
	g3 = add_mod_ntt_gpu(s_a[operation_id + t], V, bid);
	g4 = sub_mod_ntt_gpu(s_a[operation_id + t], V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + tid / t * 2];
	V = mul_mod_ntt_gpu(g3, S, bid);
	s_a[operation_id] = add_mod_ntt_gpu(g1, V, bid);
	s_a[operation_id + t] = sub_mod_ntt_gpu(g1, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + tid / t * 2 + 1];
	V = mul_mod_ntt_gpu(g4, S, bid);
	s_a[operation_id + 2 * t] = add_mod_ntt_gpu(g2, V, bid);
	s_a[operation_id + 3 * t] = sub_mod_ntt_gpu(g2, V, bid);
	__syncthreads();

	t = t / 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_gpu[bid][SIFE_N / t / 4 + tid / t];
	V = mul_mod_ntt_gpu(s_a[operation_id + 2 * t], S, bid);
	g1 = add_mod_ntt_gpu(s_a[operation_id], V, bid);
	g2 = sub_mod_ntt_gpu(s_a[operation_id], V, bid);
	V = mul_mod_ntt_gpu(s_a[operation_id + 3 * t], S, bid);
	g3 = add_mod_ntt_gpu(s_a[operation_id + t], V, bid);
	g4 = sub_mod_ntt_gpu(s_a[operation_id + t], V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + tid / t * 2];
	V = mul_mod_ntt_gpu(g3, S, bid);
	s_a[operation_id] = add_mod_ntt_gpu(g1, V, bid);
	s_a[operation_id + t] = sub_mod_ntt_gpu(g1, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + tid / t * 2 + 1];
	V = mul_mod_ntt_gpu(g4, S, bid);
	s_a[operation_id + 2 * t] = add_mod_ntt_gpu(g2, V, bid);
	s_a[operation_id + 3 * t] = sub_mod_ntt_gpu(g2, V, bid);
	__syncthreads();

	t = t / 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_gpu[bid][SIFE_N / t / 4 + tid / t];
	V = mul_mod_ntt_gpu(s_a[operation_id + 2 * t], S, bid);
	g1 = add_mod_ntt_gpu(s_a[operation_id], V, bid);
	g2 = sub_mod_ntt_gpu(s_a[operation_id], V, bid);
	V = mul_mod_ntt_gpu(s_a[operation_id + 3 * t], S, bid);
	g3 = add_mod_ntt_gpu(s_a[operation_id + t], V, bid);
	g4 = sub_mod_ntt_gpu(s_a[operation_id + t], V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + tid / t * 2];
	V = mul_mod_ntt_gpu(g3, S, bid);
	s_a[operation_id] = add_mod_ntt_gpu(g1, V, bid);
	s_a[operation_id + t] = sub_mod_ntt_gpu(g1, V, bid);
	S = psi_gpu[bid][SIFE_N / t / 2 + tid / t * 2 + 1];
	V = mul_mod_ntt_gpu(g4, S, bid);
	s_a[operation_id + 2 * t] = add_mod_ntt_gpu(g2, V, bid);
	s_a[operation_id + 3 * t] = sub_mod_ntt_gpu(g2, V, bid);
	__syncthreads();

	a[repeat*gridDim.x*SIFE_N + blockIdx.x*SIFE_N + tid] = s_a[tid];
	a[repeat*gridDim.x*SIFE_N + blockIdx.x*SIFE_N + tid + 1024] = s_a[tid + 1024];
	a[repeat*gridDim.x*SIFE_N + blockIdx.x*SIFE_N + tid + 2048] = s_a[tid + 2048];
	a[repeat*gridDim.x*SIFE_N + blockIdx.x*SIFE_N + tid + 3072] = s_a[tid + 3072];
}

// GS_reverse
// 1ºí·Ï ¹× Áö¿ªº¯¼ö È°¿ë (1¶ó¿îµåºÎÅÍ 2¶ó¿îµå¾¿ ¹­À½)
__global__ void GS_reverse_gpu_1block_2round(uint32_t a[SIFE_N]) {
	int64_t t, S, U, g1, g2, g3, g4;
	uint32_t tid, operation_id, bid = blockIdx.x%SIFE_NMODULI;
	uint32_t repeat = blockIdx.y;
	__shared__ uint32_t s_a[SIFE_N];

	tid = threadIdx.x;
	s_a[tid] = a[repeat*gridDim.x*SIFE_N + blockIdx.x*SIFE_N + tid];
	s_a[tid + 1024] = a[repeat*gridDim.x*SIFE_N + blockIdx.x*SIFE_N +tid + 1024];
	s_a[tid + 2048] = a[repeat*gridDim.x*SIFE_N + blockIdx.x*SIFE_N + tid + 2048];
	s_a[tid + 3072] = a[repeat*gridDim.x*SIFE_N + blockIdx.x*SIFE_N + tid + 3072];
	__syncthreads();

	t = 1;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2];
	U = s_a[operation_id];
	g2 = s_a[operation_id + t];
	g1 = add_mod_ntt_gpu(U, g2, bid);
	g2 = sub_mod_ntt_gpu(U, g2, bid);
	g2 = mul_mod_ntt_gpu(g2, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2 + 1];
	U = s_a[operation_id + 2 * t];
	g4 = s_a[operation_id + 3 * t];
	g3 = add_mod_ntt_gpu(U, g4, bid);
	g4 = sub_mod_ntt_gpu(U, g4, bid);
	g4 = mul_mod_ntt_gpu(g4, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + tid / t];
	U = g1;
	s_a[operation_id] = add_mod_ntt_gpu(U, g3, bid);
	g3 = sub_mod_ntt_gpu(U, g3, bid);
	s_a[operation_id + 2 * t] = mul_mod_ntt_gpu(g3, S, bid);
	U = g2;
	s_a[operation_id + t] = add_mod_ntt_gpu(U, g4, bid);
	g4 = sub_mod_ntt_gpu(U, g4, bid);
	s_a[operation_id + 3 * t] = mul_mod_ntt_gpu(g4, S, bid);
	__syncthreads();

	t = t * 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2];
	U = s_a[operation_id];
	g2 = s_a[operation_id + t];
	g1 = add_mod_ntt_gpu(U, g2, bid);
	g2 = sub_mod_ntt_gpu(U, g2, bid);
	g2 = mul_mod_ntt_gpu(g2, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2 + 1];
	U = s_a[operation_id + 2 * t];
	g4 = s_a[operation_id + 3 * t];
	g3 = add_mod_ntt_gpu(U, g4, bid);
	g4 = sub_mod_ntt_gpu(U, g4, bid);
	g4 = mul_mod_ntt_gpu(g4, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + tid / t];
	U = g1;
	s_a[operation_id] = add_mod_ntt_gpu(U, g3, bid);
	g3 = sub_mod_ntt_gpu(U, g3, bid);
	s_a[operation_id + 2 * t] = mul_mod_ntt_gpu(g3, S, bid);
	U = g2;
	s_a[operation_id + t] = add_mod_ntt_gpu(U, g4, bid);
	g4 = sub_mod_ntt_gpu(U, g4, bid);
	s_a[operation_id + 3 * t] = mul_mod_ntt_gpu(g4, S, bid);
	__syncthreads();

	t = t * 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2];
	U = s_a[operation_id];
	g2 = s_a[operation_id + t];
	g1 = add_mod_ntt_gpu(U, g2, bid);
	g2 = sub_mod_ntt_gpu(U, g2, bid);
	g2 = mul_mod_ntt_gpu(g2, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2 + 1];
	U = s_a[operation_id + 2 * t];
	g4 = s_a[operation_id + 3 * t];
	g3 = add_mod_ntt_gpu(U, g4, bid);
	g4 = sub_mod_ntt_gpu(U, g4, bid);
	g4 = mul_mod_ntt_gpu(g4, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + tid / t];
	U = g1;
	s_a[operation_id] = add_mod_ntt_gpu(U, g3, bid);
	g3 = sub_mod_ntt_gpu(U, g3, bid);
	s_a[operation_id + 2 * t] = mul_mod_ntt_gpu(g3, S, bid);
	U = g2;
	s_a[operation_id + t] = add_mod_ntt_gpu(U, g4, bid);
	g4 = sub_mod_ntt_gpu(U, g4, bid);
	s_a[operation_id + 3 * t] = mul_mod_ntt_gpu(g4, S, bid);
	__syncthreads();

	t = t * 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2];
	U = s_a[operation_id];
	g2 = s_a[operation_id + t];
	g1 = add_mod_ntt_gpu(U, g2, bid);
	g2 = sub_mod_ntt_gpu(U, g2, bid);
	g2 = mul_mod_ntt_gpu(g2, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2 + 1];
	U = s_a[operation_id + 2 * t];
	g4 = s_a[operation_id + 3 * t];
	g3 = add_mod_ntt_gpu(U, g4, bid);
	g4 = sub_mod_ntt_gpu(U, g4, bid);
	g4 = mul_mod_ntt_gpu(g4, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + tid / t];
	U = g1;
	s_a[operation_id] = add_mod_ntt_gpu(U, g3, bid);
	g3 = sub_mod_ntt_gpu(U, g3, bid);
	s_a[operation_id + 2 * t] = mul_mod_ntt_gpu(g3, S, bid);
	U = g2;
	s_a[operation_id + t] = add_mod_ntt_gpu(U, g4, bid);
	g4 = sub_mod_ntt_gpu(U, g4, bid);
	s_a[operation_id + 3 * t] = mul_mod_ntt_gpu(g4, S, bid);
	__syncthreads();

	t = t * 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2];
	U = s_a[operation_id];
	g2 = s_a[operation_id + t];
	g1 = add_mod_ntt_gpu(U, g2, bid);
	g2 = sub_mod_ntt_gpu(U, g2, bid);
	g2 = mul_mod_ntt_gpu(g2, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2 + 1];
	U = s_a[operation_id + 2 * t];
	g4 = s_a[operation_id + 3 * t];
	g3 = add_mod_ntt_gpu(U, g4, bid);
	g4 = sub_mod_ntt_gpu(U, g4, bid);
	g4 = mul_mod_ntt_gpu(g4, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + tid / t];
	U = g1;
	s_a[operation_id] = add_mod_ntt_gpu(U, g3, bid);
	g3 = sub_mod_ntt_gpu(U, g3, bid);
	s_a[operation_id + 2 * t] = mul_mod_ntt_gpu(g3, S, bid);
	U = g2;
	s_a[operation_id + t] = add_mod_ntt_gpu(U, g4, bid);
	g4 = sub_mod_ntt_gpu(U, g4, bid);
	s_a[operation_id + 3 * t] = mul_mod_ntt_gpu(g4, S, bid);
	__syncthreads();

	t = t * 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2];
	U = s_a[operation_id];
	g2 = s_a[operation_id + t];
	g1 = add_mod_ntt_gpu(U, g2, bid);
	g2 = sub_mod_ntt_gpu(U, g2, bid);
	g2 = mul_mod_ntt_gpu(g2, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2 + 1];
	U = s_a[operation_id + 2 * t];
	g4 = s_a[operation_id + 3 * t];
	g3 = add_mod_ntt_gpu(U, g4, bid);
	g4 = sub_mod_ntt_gpu(U, g4, bid);
	g4 = mul_mod_ntt_gpu(g4, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + tid / t];
	U = g1;
	s_a[operation_id] = add_mod_ntt_gpu(U, g3, bid);
	g3 = sub_mod_ntt_gpu(U, g3, bid);
	s_a[operation_id + 2 * t] = mul_mod_ntt_gpu(g3, S, bid);
	U = g2;
	s_a[operation_id + t] = add_mod_ntt_gpu(U, g4, bid);
	g4 = sub_mod_ntt_gpu(U, g4, bid);
	s_a[operation_id + 3 * t] = mul_mod_ntt_gpu(g4, S, bid);
	__syncthreads();

	a[repeat*gridDim.x*SIFE_N + blockIdx.x*SIFE_N + tid] = mul_mod_ntt_gpu(s_a[tid], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*gridDim.x*SIFE_N + blockIdx.x*SIFE_N + tid + 1024] = mul_mod_ntt_gpu(s_a[tid + 1024], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*gridDim.x*SIFE_N + blockIdx.x*SIFE_N + tid + 2048] = mul_mod_ntt_gpu(s_a[tid + 2048], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*gridDim.x*SIFE_N + blockIdx.x*SIFE_N + tid + 3072] = mul_mod_ntt_gpu(s_a[tid + 3072], SIFE_NTT_NINV_gpu[bid], bid);
}


void CT_forward_Moduli_GPU(uint32_t a[SIFE_NMODULI][SIFE_N])
{
	uint32_t *d_a;
	
	cudaMalloc((void**)&d_a, SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMemcpy(d_a, a, SIFE_NMODULI*SIFE_N*sizeof(uint32_t),cudaMemcpyHostToDevice);

	CT_forward_gpu_1block_1round << <SIFE_NMODULI, 1024 >> > (d_a);

	cudaMemcpy(a, d_a, SIFE_NMODULI*SIFE_N*sizeof(uint32_t), cudaMemcpyDeviceToHost);

	cudaFree(d_a);
}

void CT_forward_Moduli_L_GPU(uint32_t *a)
{
	uint32_t *d_a;
	
	cudaMalloc((void**)&d_a, SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMemcpy(d_a, a, SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t),cudaMemcpyHostToDevice);

	CT_forward_gpu_1block_1round << <SIFE_L*SIFE_NMODULI, 1024 >> > (d_a);

	cudaMemcpy(a, d_a, SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t), cudaMemcpyDeviceToHost);

	cudaFree(d_a);
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

	//point_mul_gpu2
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



//NTT 함수 테스트 중
__global__ void CT_forward_gpu_64block_2kernel_1round_1_batch(uint32_t* a) {
	int64_t t, S, V;
	uint32_t thread_id, operation_id;
	__shared__ uint32_t s_a[SIFE_N / 64];
	uint32_t  sel = (blockIdx.x/64)%SIFE_NMODULI;
	uint32_t  bid = blockIdx.x % 64;
	uint32_t repeat = blockIdx.y;

	thread_id = (threadIdx.x / 1) * 64 + threadIdx.x % 1 + bid * 1;
	s_a[threadIdx.x] = a[repeat*SIFE_NMODULI*SIFE_N + thread_id + sel*SIFE_N];
	s_a[threadIdx.x + 32] = a[repeat*SIFE_NMODULI*SIFE_N + thread_id + 2048 + sel*SIFE_N];
	__syncthreads();

	t = SIFE_N / 2;
	operation_id = threadIdx.x;
	S = psi_gpu[sel][SIFE_N / t / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + 32], S, sel);
	s_a[operation_id + 32] = sub_mod_ntt_gpu(s_a[operation_id], V, sel);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, sel);
	__syncthreads();

	t = t / 2;
	operation_id = (threadIdx.x / 16) * 32 + threadIdx.x % 16;
	S = psi_gpu[sel][SIFE_N / t / 2 + operation_id / 16 / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + 16], S, sel);
	s_a[operation_id + 16] = sub_mod_ntt_gpu(s_a[operation_id], V, sel);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, sel);
	__syncthreads();

	t = t / 2;
	operation_id = (threadIdx.x / 8) * 16 + threadIdx.x % 8;
	S = psi_gpu[sel][SIFE_N / t / 2 + operation_id / 8 / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + 8], S, sel);
	s_a[operation_id + 8] = sub_mod_ntt_gpu(s_a[operation_id], V, sel);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, sel);
	__syncthreads();

	t = t / 2;
	operation_id = (threadIdx.x / 4) * 8 + threadIdx.x % 4;
	S = psi_gpu[sel][SIFE_N / t / 2 + operation_id / 4 / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + 4], S, sel);
	s_a[operation_id + 4] = sub_mod_ntt_gpu(s_a[operation_id], V, sel);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, sel);
	__syncthreads();

	t = t / 2;
	operation_id = (threadIdx.x / 2) * 4 + threadIdx.x % 2;
	S = psi_gpu[sel][SIFE_N / t / 2 + operation_id / 2 / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + 2], S, sel);
	s_a[operation_id + 2] = sub_mod_ntt_gpu(s_a[operation_id], V, sel);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, sel);
	__syncthreads();

	t = t / 2;
	operation_id = (threadIdx.x / 1) * 2 + threadIdx.x % 1;
	S = psi_gpu[sel][SIFE_N / t / 2 + operation_id / 1 / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + 1], S, sel);
	s_a[operation_id + 1] = sub_mod_ntt_gpu(s_a[operation_id], V, sel);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, sel);
	__syncthreads();

	a[repeat*SIFE_NMODULI*SIFE_N + thread_id + sel*SIFE_N] = s_a[threadIdx.x];
	a[repeat*SIFE_NMODULI*SIFE_N + thread_id + 2048 + sel*SIFE_N] = s_a[threadIdx.x + 32];
}
__global__ void CT_forward_gpu_64block_2kernel_1round_2_batch(uint32_t* a) {
	int64_t t, S, V;
	uint32_t thread_id, operation_id, period;
	__shared__ uint32_t s_a[SIFE_N / 64];
	uint32_t  sel = (blockIdx.x/64)%SIFE_NMODULI;
	uint32_t  bid = blockIdx.x % 64;
	uint32_t repeat = blockIdx.y;

	period = SIFE_N / (gridDim.x / SIFE_NMODULI);
	t = period;
	thread_id = threadIdx.x + bid * period;
	s_a[threadIdx.x] = a[repeat*SIFE_NMODULI*SIFE_N + thread_id + sel*SIFE_N];
	s_a[threadIdx.x + period / 2] = a[repeat*SIFE_NMODULI*SIFE_N + thread_id + period / 2 + sel*SIFE_N];
	__syncthreads();

	t = t / 2;
	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(s_a[operation_id], V, sel);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, sel);
	__syncthreads();

	t = t / 2;
	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(s_a[operation_id], V, sel);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, sel);
	__syncthreads();

	t = t / 2;
	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(s_a[operation_id], V, sel);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, sel);
	__syncthreads();

	t = t / 2;
	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(s_a[operation_id], V, sel);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, sel);
	__syncthreads();

	t = t / 2;
	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(s_a[operation_id], V, sel);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, sel);
	__syncthreads();

	t = t / 2;
	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	V = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(s_a[operation_id], V, sel);
	s_a[operation_id] = add_mod_ntt_gpu(s_a[operation_id], V, sel);
	__syncthreads();

	a[repeat*SIFE_NMODULI*SIFE_N + thread_id + sel*SIFE_N] = s_a[threadIdx.x];
	a[repeat*SIFE_NMODULI*SIFE_N + thread_id + period / 2 + sel*SIFE_N] = s_a[threadIdx.x + period / 2];
}
__global__ void GS_reverse_gpu_64block_2kernel_1round_1_batch(uint32_t* a) {
	int64_t t, S, U;
	uint32_t thread_id, operation_id, period;
	__shared__ uint32_t s_a[SIFE_N / 64];
	uint32_t  sel = (blockIdx.x/64)%SIFE_NMODULI;
	uint32_t  bid = blockIdx.x % 64;
	uint32_t repeat = blockIdx.y;

	period = SIFE_N / (gridDim.x / SIFE_NMODULI);
	thread_id = threadIdx.x + bid * period;
	s_a[threadIdx.x] = a[repeat*gridDim.x/64*blockDim.x*2 + thread_id + sel * SIFE_N];
	s_a[threadIdx.x + period / 2] = a[repeat*gridDim.x/64*blockDim.x*2 + thread_id + period / 2 + sel * SIFE_N];
	__syncthreads();

	t = 1;
	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	U = s_a[operation_id];
	s_a[operation_id] = add_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	U = s_a[operation_id];
	s_a[operation_id] = add_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	U = s_a[operation_id];
	s_a[operation_id] = add_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	U = s_a[operation_id];
	s_a[operation_id] = add_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	U = s_a[operation_id];
	s_a[operation_id] = add_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	U = s_a[operation_id];
	s_a[operation_id] = add_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	__syncthreads();

	a[repeat*gridDim.x/64*blockDim.x*2 + thread_id + sel * SIFE_N] = s_a[threadIdx.x];
	a[repeat*gridDim.x/64*blockDim.x*2 + thread_id + period / 2 + sel * SIFE_N] = s_a[threadIdx.x + period / 2];
}
__global__ void GS_reverse_gpu_64block_2kernel_1round_2_batch(uint32_t* a) {
	int64_t t, S, U, V, g0, g1;
	uint32_t thread_id, operation_id;
	__shared__ uint32_t s_a[SIFE_N / 64];
	uint32_t  sel = (blockIdx.x/64)%SIFE_NMODULI;
	uint32_t  bid = blockIdx.x % 64;
	uint32_t repeat = blockIdx.y;

	t = 64;
	thread_id = (threadIdx.x / 1) * 64 + threadIdx.x % 1 + bid * 1;
	s_a[threadIdx.x] = a[repeat*gridDim.x/64*blockDim.x*2 + thread_id + sel * SIFE_N];
	s_a[threadIdx.x + 32] = a[repeat*gridDim.x/64*blockDim.x*2 + thread_id + 2048 + sel * SIFE_N];
	__syncthreads();

	operation_id = (threadIdx.x / 1) * 2 + threadIdx.x % 1;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + operation_id / 1 / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + 1];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, sel);
	V = sub_mod_ntt_gpu(U, V, sel);
	s_a[operation_id + 1] = mul_mod_ntt_gpu(V, S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / 2) * 4 + threadIdx.x % 2;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + operation_id / 2 / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + 2];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, sel);
	V = sub_mod_ntt_gpu(U, V, sel);
	s_a[operation_id + 2] = mul_mod_ntt_gpu(V, S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / 4) * 8 + threadIdx.x % 4;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + operation_id / 4 / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + 4];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, sel);
	V = sub_mod_ntt_gpu(U, V, sel);
	s_a[operation_id + 4] = mul_mod_ntt_gpu(V, S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / 8) * 16 + threadIdx.x % 8;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + operation_id / 8 / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + 8];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, sel);
	V = sub_mod_ntt_gpu(U, V, sel);
	s_a[operation_id + 8] = mul_mod_ntt_gpu(V, S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / 16) * 32 + threadIdx.x % 16;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + operation_id / 16 / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + 16];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, sel);
	V = sub_mod_ntt_gpu(U, V, sel);
	s_a[operation_id + 16] = mul_mod_ntt_gpu(V, S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = threadIdx.x;
	S = psi_inv_gpu[sel][SIFE_N / t / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + 32];
	g0 = add_mod_ntt_gpu(U, V, sel);
	V = sub_mod_ntt_gpu(U, V, sel);
	g1 = mul_mod_ntt_gpu(V, S, sel);
	__syncthreads();

	a[repeat*gridDim.x/64*blockDim.x*2 + thread_id + sel * SIFE_N] = mul_mod_ntt_gpu(g0, SIFE_NTT_NINV_gpu[sel], sel);
	a[repeat*gridDim.x/64*blockDim.x*2 + thread_id + 2048 + sel * SIFE_N] = mul_mod_ntt_gpu(g1, SIFE_NTT_NINV_gpu[sel], sel);
}
__global__ void GS_reverse_gpu_64block_2kernel_1round_1_batch2(uint32_t* a) {
	int64_t t, S, U;
	uint32_t thread_id, operation_id, period;
	__shared__ uint32_t s_a[SIFE_N / 64];
	uint32_t  sel = (blockIdx.x/64)%SIFE_NMODULI;
	uint32_t  bid = blockIdx.x % 64;
	uint32_t repeat = blockIdx.y;

	period = SIFE_N / 64;
	thread_id = threadIdx.x + bid * period;
	s_a[threadIdx.x] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + (blockIdx.x/64/SIFE_NMODULI)*SIFE_NMODULI*SIFE_N + thread_id + sel * SIFE_N];
	s_a[threadIdx.x + period / 2] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + (blockIdx.x/64/SIFE_NMODULI)*SIFE_NMODULI*SIFE_N + thread_id + period / 2 + sel * SIFE_N];
	__syncthreads();

	t = 1;
	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	U = s_a[operation_id];
	s_a[operation_id] = add_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	U = s_a[operation_id];
	s_a[operation_id] = add_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	U = s_a[operation_id];
	s_a[operation_id] = add_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	U = s_a[operation_id];
	s_a[operation_id] = add_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	U = s_a[operation_id];
	s_a[operation_id] = add_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	U = s_a[operation_id];
	s_a[operation_id] = add_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	__syncthreads();

	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + (blockIdx.x/64/SIFE_NMODULI)*SIFE_NMODULI*SIFE_N + thread_id + sel * SIFE_N] = s_a[threadIdx.x];
	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + (blockIdx.x/64/SIFE_NMODULI)*SIFE_NMODULI*SIFE_N + thread_id + period / 2 + sel * SIFE_N] = s_a[threadIdx.x + period / 2];
}
__global__ void GS_reverse_gpu_64block_2kernel_1round_2_batch2(uint32_t* a) {
	int64_t t, S, U, V, g0, g1;
	uint32_t thread_id, operation_id;
	__shared__ uint32_t s_a[SIFE_N / 64];
	uint32_t  sel = (blockIdx.x/64)%SIFE_NMODULI;
	uint32_t  bid = blockIdx.x % 64;
	uint32_t repeat = blockIdx.y;

	t = 64;
	thread_id = (threadIdx.x / 1) * 64 + threadIdx.x % 1 + bid * 1;
	s_a[threadIdx.x] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + (blockIdx.x/64/SIFE_NMODULI)*SIFE_NMODULI*SIFE_N + thread_id + sel * SIFE_N];
	s_a[threadIdx.x + 32] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + (blockIdx.x/64/SIFE_NMODULI)*SIFE_NMODULI*SIFE_N + thread_id + 2048 + sel * SIFE_N];
	__syncthreads();

	operation_id = (threadIdx.x / 1) * 2 + threadIdx.x % 1;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + operation_id / 1 / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + 1];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, sel);
	V = sub_mod_ntt_gpu(U, V, sel);
	s_a[operation_id + 1] = mul_mod_ntt_gpu(V, S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / 2) * 4 + threadIdx.x % 2;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + operation_id / 2 / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + 2];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, sel);
	V = sub_mod_ntt_gpu(U, V, sel);
	s_a[operation_id + 2] = mul_mod_ntt_gpu(V, S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / 4) * 8 + threadIdx.x % 4;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + operation_id / 4 / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + 4];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, sel);
	V = sub_mod_ntt_gpu(U, V, sel);
	s_a[operation_id + 4] = mul_mod_ntt_gpu(V, S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / 8) * 16 + threadIdx.x % 8;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + operation_id / 8 / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + 8];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, sel);
	V = sub_mod_ntt_gpu(U, V, sel);
	s_a[operation_id + 8] = mul_mod_ntt_gpu(V, S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / 16) * 32 + threadIdx.x % 16;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + operation_id / 16 / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + 16];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, sel);
	V = sub_mod_ntt_gpu(U, V, sel);
	s_a[operation_id + 16] = mul_mod_ntt_gpu(V, S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = threadIdx.x;
	S = psi_inv_gpu[sel][SIFE_N / t / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + 32];
	g0 = add_mod_ntt_gpu(U, V, sel);
	V = sub_mod_ntt_gpu(U, V, sel);
	g1 = mul_mod_ntt_gpu(V, S, sel);
	__syncthreads();

	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + (blockIdx.x/64/SIFE_NMODULI)*SIFE_NMODULI*SIFE_N + thread_id + sel * SIFE_N] = mul_mod_ntt_gpu(g0, SIFE_NTT_NINV_gpu[sel], sel);
	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + (blockIdx.x/64/SIFE_NMODULI)*SIFE_NMODULI*SIFE_N + thread_id + 2048 + sel * SIFE_N] = mul_mod_ntt_gpu(g1, SIFE_NTT_NINV_gpu[sel], sel);
}

__global__ void GS_reverse_gpu_64block_2kernel_1round_1_batch3(uint32_t* a) {
	int64_t t, S, U;
	uint32_t thread_id, operation_id, period;
	__shared__ uint32_t s_a[SIFE_N / 64];
	uint32_t  sel = (blockIdx.x/64)%SIFE_NMODULI;
	uint32_t  bid = blockIdx.x % 64;
	uint32_t repeat = blockIdx.y;

	period = SIFE_N / 64;
	thread_id = threadIdx.x + bid * period;
	s_a[threadIdx.x] = a[repeat*SIFE_NMODULI*SIFE_N + thread_id + sel * SIFE_N];
	s_a[threadIdx.x + period / 2] = a[repeat*SIFE_NMODULI*SIFE_N + thread_id + period / 2 + sel * SIFE_N];
	__syncthreads();

	t = 1;
	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	U = s_a[operation_id];
	s_a[operation_id] = add_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	U = s_a[operation_id];
	s_a[operation_id] = add_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	U = s_a[operation_id];
	s_a[operation_id] = add_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	U = s_a[operation_id];
	s_a[operation_id] = add_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	U = s_a[operation_id];
	s_a[operation_id] = add_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / t) * t * 2 + threadIdx.x % t;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + (operation_id + bid * period) / t / 2];
	U = s_a[operation_id];
	s_a[operation_id] = add_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = sub_mod_ntt_gpu(U, s_a[operation_id + t], sel);
	s_a[operation_id + t] = mul_mod_ntt_gpu(s_a[operation_id + t], S, sel);
	__syncthreads();

	a[repeat*SIFE_NMODULI*SIFE_N + thread_id + sel * SIFE_N] = s_a[threadIdx.x];
	a[repeat*SIFE_NMODULI*SIFE_N + thread_id + period / 2 + sel * SIFE_N] = s_a[threadIdx.x + period / 2];
}
__global__ void GS_reverse_gpu_64block_2kernel_1round_2_batch3(uint32_t* a) {
	int64_t t, S, U, V, g0, g1;
	uint32_t thread_id, operation_id;
	__shared__ uint32_t s_a[SIFE_N / 64];
	uint32_t  sel = (blockIdx.x/64)%SIFE_NMODULI;
	uint32_t  bid = blockIdx.x % 64;
	uint32_t repeat = blockIdx.y;

	t = 64;
	thread_id = (threadIdx.x / 1) * 64 + threadIdx.x % 1 + bid * 1;
	s_a[threadIdx.x] = a[repeat*SIFE_NMODULI*SIFE_N + thread_id + sel * SIFE_N];
	s_a[threadIdx.x + 32] = a[repeat*SIFE_NMODULI*SIFE_N + thread_id + 2048 + sel * SIFE_N];
	__syncthreads();

	operation_id = (threadIdx.x / 1) * 2 + threadIdx.x % 1;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + operation_id / 1 / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + 1];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, sel);
	V = sub_mod_ntt_gpu(U, V, sel);
	s_a[operation_id + 1] = mul_mod_ntt_gpu(V, S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / 2) * 4 + threadIdx.x % 2;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + operation_id / 2 / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + 2];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, sel);
	V = sub_mod_ntt_gpu(U, V, sel);
	s_a[operation_id + 2] = mul_mod_ntt_gpu(V, S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / 4) * 8 + threadIdx.x % 4;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + operation_id / 4 / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + 4];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, sel);
	V = sub_mod_ntt_gpu(U, V, sel);
	s_a[operation_id + 4] = mul_mod_ntt_gpu(V, S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / 8) * 16 + threadIdx.x % 8;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + operation_id / 8 / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + 8];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, sel);
	V = sub_mod_ntt_gpu(U, V, sel);
	s_a[operation_id + 8] = mul_mod_ntt_gpu(V, S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = (threadIdx.x / 16) * 32 + threadIdx.x % 16;
	S = psi_inv_gpu[sel][SIFE_N / t / 2 + operation_id / 16 / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + 16];
	s_a[operation_id] = add_mod_ntt_gpu(U, V, sel);
	V = sub_mod_ntt_gpu(U, V, sel);
	s_a[operation_id + 16] = mul_mod_ntt_gpu(V, S, sel);
	t = t * 2;
	__syncthreads();

	operation_id = threadIdx.x;
	S = psi_inv_gpu[sel][SIFE_N / t / 2];
	U = s_a[operation_id];
	V = s_a[operation_id + 32];
	g0 = add_mod_ntt_gpu(U, V, sel);
	V = sub_mod_ntt_gpu(U, V, sel);
	g1 = mul_mod_ntt_gpu(V, S, sel);
	__syncthreads();

	a[repeat*SIFE_NMODULI*SIFE_N + thread_id + sel * SIFE_N] = mul_mod_ntt_gpu(g0, SIFE_NTT_NINV_gpu[sel], sel);
	a[repeat*SIFE_NMODULI*SIFE_N + thread_id + 2048 + sel * SIFE_N] = mul_mod_ntt_gpu(g1, SIFE_NTT_NINV_gpu[sel], sel);
}

__global__ void CT_forward_gpu_1block_3round(uint32_t* a) {
	int64_t t, S, V, g0, g1, g2, g3, g4, g5, g6, g7;
	uint32_t thread_id, operation_id;
	__shared__ uint32_t s_a[SIFE_N];
	uint32_t bid = blockIdx.x%SIFE_NMODULI;
	uint32_t repeat = blockIdx.y;

	t = SIFE_N;
	thread_id = threadIdx.x;
	s_a[thread_id] = a[repeat*gridDim.x*SIFE_N + thread_id + bid * SIFE_N];
	s_a[thread_id + 512] = a[repeat*gridDim.x*SIFE_N + thread_id + 512 + bid * SIFE_N];
	s_a[thread_id + 1024] = a[repeat*gridDim.x*SIFE_N + thread_id + 1024 + bid * SIFE_N];
	s_a[thread_id + 1536] = a[repeat*gridDim.x*SIFE_N + thread_id + 1536 + bid * SIFE_N];
	s_a[thread_id + 2048] = a[repeat*gridDim.x*SIFE_N + thread_id + 2048 + bid * SIFE_N];
	s_a[thread_id + 2560] = a[repeat*gridDim.x*SIFE_N + thread_id + 2560 + bid * SIFE_N];
	s_a[thread_id + 3072] = a[repeat*gridDim.x*SIFE_N + thread_id + 3072 + bid * SIFE_N];
	s_a[thread_id + 3584] = a[repeat*gridDim.x*SIFE_N + thread_id + 3584 + bid * SIFE_N];

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

	a[repeat*gridDim.x*SIFE_N + thread_id + bid * SIFE_N] = s_a[thread_id];
	a[repeat*gridDim.x*SIFE_N + thread_id + 512 + bid * SIFE_N] = s_a[thread_id + 512];
	a[repeat*gridDim.x*SIFE_N + thread_id + 1024 + bid * SIFE_N] = s_a[thread_id + 1024];
	a[repeat*gridDim.x*SIFE_N + thread_id + 1536 + bid * SIFE_N] = s_a[thread_id + 1536];
	a[repeat*gridDim.x*SIFE_N + thread_id + 2048 + bid * SIFE_N] = s_a[thread_id + 2048];
	a[repeat*gridDim.x*SIFE_N + thread_id + 2560 + bid * SIFE_N] = s_a[thread_id + 2560];
	a[repeat*gridDim.x*SIFE_N + thread_id + 3072 + bid * SIFE_N] = s_a[thread_id + 3072];
	a[repeat*gridDim.x*SIFE_N + thread_id + 3584 + bid * SIFE_N] = s_a[thread_id + 3584];
}
__global__ void GS_reverse_gpu_1block_3round(uint32_t* a) {
	int64_t t, S, U, g0, g1, g2, g3, g4, g5, g6, g7;
	uint32_t thread_id, operation_id;
	__shared__ uint32_t s_a[SIFE_N];
	uint32_t bid = blockIdx.x%SIFE_NMODULI;
	uint32_t repeat = blockIdx.y;

	thread_id = threadIdx.x;
	s_a[thread_id] = a[repeat*gridDim.x*SIFE_N + thread_id + bid * SIFE_N];
	s_a[thread_id + 512] = a[repeat*gridDim.x*SIFE_N + thread_id + 512 + bid * SIFE_N];
	s_a[thread_id + 1024] = a[repeat*gridDim.x*SIFE_N + thread_id + 1024 + bid * SIFE_N];
	s_a[thread_id + 1536] = a[repeat*gridDim.x*SIFE_N + thread_id + 1536 + bid * SIFE_N];
	s_a[thread_id + 2048] = a[repeat*gridDim.x*SIFE_N + thread_id + 2048 + bid * SIFE_N];
	s_a[thread_id + 2560] = a[repeat*gridDim.x*SIFE_N + thread_id + 2560 + bid * SIFE_N];
	s_a[thread_id + 3072] = a[repeat*gridDim.x*SIFE_N + thread_id + 3072 + bid * SIFE_N];
	s_a[thread_id + 3584] = a[repeat*gridDim.x*SIFE_N + thread_id + 3584 + bid * SIFE_N];
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

	a[repeat*gridDim.x*SIFE_N + thread_id + bid * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*gridDim.x*SIFE_N + thread_id + 512 + bid * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id + 512], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*gridDim.x*SIFE_N + thread_id + 1024 + bid * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id + 1024], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*gridDim.x*SIFE_N + thread_id + 1536 + bid * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id + 1536], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*gridDim.x*SIFE_N + thread_id + 2048 + bid * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id + 2048], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*gridDim.x*SIFE_N + thread_id + 2560 + bid * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id + 2560], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*gridDim.x*SIFE_N + thread_id + 3072 + bid * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id + 3072], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*gridDim.x*SIFE_N + thread_id + 3584 + bid * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id + 3584], SIFE_NTT_NINV_gpu[bid], bid);
}
