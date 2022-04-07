
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdint.h>

#include "params.h"
#include "AES.cuh"
#include "sampler.cuh"
#include "arith_rns.cuh"
#include "randombytes.h"
#include "ntt_gpu.cuh"
#include "consts.cuh"


__device__ uint32_t add_mod_ntt_gpu_2(uint32_t a, uint32_t b, uint32_t sel)
{
	uint64_t c;

	c = (uint64_t)a + (uint64_t)b;

	if (c >= SIFE_MOD_Q_I_gpu[sel]) {
		c -= SIFE_MOD_Q_I_gpu[sel];
	}
	return (uint32_t)c;
}

__device__ uint32_t sub_mod_ntt_gpu_2(uint32_t a, uint32_t b, uint32_t sel)//returns a-b Mod Q
{
	uint64_t c;

	c = (uint64_t)a + (uint64_t)SIFE_MOD_Q_I_gpu[sel] - (uint64_t)b;

	if (c >= SIFE_MOD_Q_I_gpu[sel]) {
		c -= SIFE_MOD_Q_I_gpu[sel];
	}
	return (uint32_t)c;
}

__global__ void point_mul_gpu_2(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b)
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

__global__ void point_mul_gpu2_2(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b)
{
	uint32_t tid = threadIdx.x, bid = blockIdx.x*SIFE_N;
	uint32_t repeat = blockIdx.y;
	int32_t i;
	for (i = 0; i < SIFE_N/1024; ++i)
	{
		d_c[repeat*(SIFE_L+1)*gridDim.x*SIFE_N + bid+tid+i*1024] = mul_mod_ntt_gpu(d_a[repeat*gridDim.x*SIFE_N + bid + tid+i*1024], d_b[bid+tid+i*1024], blockIdx.x);			
	}	
}
__global__ void point_mul_gpu2_3(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b)
{
	uint32_t tid = threadIdx.x, bid = blockIdx.x*SIFE_N;
	uint32_t repeat = blockIdx.y;
	int32_t i;
	for (i = 0; i < SIFE_N/1024; ++i)
	{
		d_c[repeat*gridDim.x*SIFE_N + bid+tid+i*1024] = mul_mod_ntt_gpu(d_a[repeat*gridDim.x*SIFE_N + bid + tid+i*1024], d_b[bid+tid+i*1024], blockIdx.x);			
	}	
}

__global__ void point_add_mod_gpu2_2(uint32_t *d_c, uint32_t *d_m)
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

__global__ void point_add_mod_gpu3_2(uint32_t *d_c, uint32_t *d_a, uint32_t *d_b)
{
	uint32_t tid = threadIdx.x, bid = blockIdx.x*SIFE_N;
	uint32_t repeat = blockIdx.y;
	int32_t i;
	for (i = 0; i < SIFE_N/1024; ++i)
	{
		d_c[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + bid+tid+i*1024] = add_mod_ntt_gpu_2(d_a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + bid+tid+i*1024], d_b[repeat*SIFE_NMODULI*SIFE_N + bid+tid+i*1024], blockIdx.x);
	}	
}

__global__ void poly_sub_mod_gpu2(uint32_t *a, uint32_t *b, uint32_t *c)
{
	uint32_t tid = threadIdx.x, bid = blockIdx.x*SIFE_N, i;
	uint32_t repeat = blockIdx.y;
	for (i = 0; i < SIFE_N/1024; ++i)
	{
		c[repeat*SIFE_NMODULI*SIFE_N + bid + tid+i*1024] = sub_mod_ntt_gpu_2(a[repeat*SIFE_NMODULI*SIFE_N + bid + tid+i*1024], b[repeat*SIFE_NMODULI*SIFE_N + bid + tid+i*1024], blockIdx.x);
	}
}

__global__ void gaussian_sampler_S3_gpu2(uint8_t *rk, uint32_t *d_c)
{	
	uint64_t vx64[4] = {0}, vb_in_64[4] ={0};	
	uint64_t z[8] __attribute__ ((aligned (32)));
	uint64_t b[8] __attribute__ ((aligned (32)));
	uint64_t v64_y1[8] __attribute__ ((aligned (32))) = {0};
	uint64_t v64_y2[8] __attribute__ ((aligned (32))) = {0};
	const uint32_t AES_ROUNDS=3;
	uint32_t i = 8, j = 0, l = 0;
	uint64_t k;//, start_k, stop_k;
	uint8_t *r1;
	uint64_t mod;
	uint32_t mod1, mod2, mod3;
	uint32_t tid = threadIdx.x, bid = blockIdx.x;
	uint32_t repeat = blockIdx.y;
	uint8_t r[384] = {0};
	uint32_t rep = 0;// Count no. of AES samp. done in each thread

	uint32_t sample_0, sample_1, sample_2, sample_3;

#if SEC_LEVEL==2
	uint32_t mod4;
#endif
	if (tid < 512)
	{
		while (j < LEN_THREAD)// not adjustable now, one loop 3 samples.
		{
			do
			{			
				if (i == 8)
				{
					for(l=0; l<4; l++) vx64[l] = 0;				
					aes256ctr_squeezeblocks_gpu (r, AES_ROUNDS, (uint32_t*)rk+ repeat*4*60, rep);
					uniform_sampler_S3_gpu(r + 2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE), v64_y1, v64_y2);

					r1 = r;
					cdt_sampler_gpu(r1, vx64);				    			
					for(l=0; l<4; l++) vx64[l] = (uint32_t) vx64[l] * BINARY_SAMPLER_K_S3;
					for(l=0; l<4; l++) z[l] = vx64[l] + v64_y1[l];	
					for(l=0; l<4; l++) vb_in_64[l] = z[l] + vx64[l];
					for(l=0; l<4; l++) vb_in_64[l] = (uint32_t) vb_in_64[l] * v64_y1[l];
					bernoulli_sampler_S3_gpu(b, vb_in_64, r1 + BASE_TABLE_SIZE);

					r1 = r + BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE;
					for(l=0; l<4; l++) vx64[l] = 0;
					cdt_sampler_gpu(r1, vx64);	
					for(l=0; l<4; l++) vx64[l] = (uint32_t) vx64[l] * BINARY_SAMPLER_K_S3;
					for(l=0; l<4; l++) z[l+4] = vx64[l] + v64_y2[l];
					for(l=0; l<4; l++) vb_in_64[l] = z[l+4] + vx64[l];
					for(l=0; l<4; l++) vb_in_64[l] = (uint32_t) vb_in_64[l] * v64_y2[l];
					bernoulli_sampler_S3_gpu(b + 4, vb_in_64, r1 + BASE_TABLE_SIZE);
					i = 0;
					rep++;
				}
				k = (r[2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE) + UNIFORM_REJ * UNIFORM_SIZE] >> i) & 0x1;		
				i++;
				// printf("%u %u %u b: %lu\n", tid, i, j, b[i - 1]);
			} while (1 ^ ((b[i - 1] & ((z[i - 1] | -z[i - 1]) | (k | -k))) >> 63)); /* rejection condition: b=0 or ((b=1) && (z=0) && (k=0)) */
			mod=z[i-1];

			mod1=mod_prime_gpu(mod, 0);
			mod2=mod_prime_gpu(mod, 1);
			mod3=mod_prime_gpu(mod, 2);		

			sample_0=(1-k)*mod1+k*mod_prime_gpu(SIFE_MOD_Q_I_GPU[0]-mod1, 0);
			sample_1=(1-k)*mod2+k*mod_prime_gpu(SIFE_MOD_Q_I_GPU[1]-mod2, 1);
			sample_2=(1-k)*mod3+k*mod_prime_gpu(SIFE_MOD_Q_I_GPU[2]-mod3, 2);
			#if SEC_LEVEL==2
				mod4=mod_prime_gpu(mod, 3);	
				sample_3=(1-k)*mod4+k*mod_prime_gpu(SIFE_MOD_Q_I_GPU[3]-mod4, 3);
			#endif

			d_c[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + bid*SIFE_NMODULI*SIFE_N+j+0*SIFE_N+tid*LEN_THREAD] = add_mod_ntt_gpu_2(d_c[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + bid*SIFE_NMODULI*SIFE_N+j+0*SIFE_N+tid*LEN_THREAD], sample_0, 0);
			d_c[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + bid*SIFE_NMODULI*SIFE_N+j+1*SIFE_N+tid*LEN_THREAD] = add_mod_ntt_gpu_2(d_c[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + bid*SIFE_NMODULI*SIFE_N+j+1*SIFE_N+tid*LEN_THREAD], sample_1, 1);
			d_c[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + bid*SIFE_NMODULI*SIFE_N+j+2*SIFE_N+tid*LEN_THREAD] = add_mod_ntt_gpu_2(d_c[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + bid*SIFE_NMODULI*SIFE_N+j+2*SIFE_N+tid*LEN_THREAD], sample_2, 2);
			d_c[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + bid*SIFE_NMODULI*SIFE_N+j+3*SIFE_N+tid*LEN_THREAD] = add_mod_ntt_gpu_2(d_c[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + bid*SIFE_NMODULI*SIFE_N+j+3*SIFE_N+tid*LEN_THREAD], sample_3, 3);

			j++;
		}
	}
}


__global__ void GS_reverse_gpu_1block_2round2(uint32_t a[SIFE_N]) {
	int64_t t, S, U, g1, g2, g3, g4;
	uint32_t tid, operation_id, bid = blockIdx.x%SIFE_NMODULI;
	uint32_t repeat = blockIdx.y;
	__shared__ uint32_t s_a[SIFE_N];

	tid = threadIdx.x;
	s_a[tid] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + blockIdx.x*SIFE_N + tid];
	s_a[tid + 1024] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + blockIdx.x*SIFE_N +tid + 1024];
	s_a[tid + 2048] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + blockIdx.x*SIFE_N + tid + 2048];
	s_a[tid + 3072] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + blockIdx.x*SIFE_N + tid + 3072];
	__syncthreads();

	t = 1;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2];
	U = s_a[operation_id];
	g2 = s_a[operation_id + t];
	g1 = add_mod_ntt_gpu_2(U, g2, bid);
	g2 = sub_mod_ntt_gpu_2(U, g2, bid);
	g2 = mul_mod_ntt_gpu(g2, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2 + 1];
	U = s_a[operation_id + 2 * t];
	g4 = s_a[operation_id + 3 * t];
	g3 = add_mod_ntt_gpu_2(U, g4, bid);
	g4 = sub_mod_ntt_gpu_2(U, g4, bid);
	g4 = mul_mod_ntt_gpu(g4, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + tid / t];
	U = g1;
	s_a[operation_id] = add_mod_ntt_gpu_2(U, g3, bid);
	g3 = sub_mod_ntt_gpu_2(U, g3, bid);
	s_a[operation_id + 2 * t] = mul_mod_ntt_gpu(g3, S, bid);
	U = g2;
	s_a[operation_id + t] = add_mod_ntt_gpu_2(U, g4, bid);
	g4 = sub_mod_ntt_gpu_2(U, g4, bid);
	s_a[operation_id + 3 * t] = mul_mod_ntt_gpu(g4, S, bid);
	__syncthreads();

	t = t * 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2];
	U = s_a[operation_id];
	g2 = s_a[operation_id + t];
	g1 = add_mod_ntt_gpu_2(U, g2, bid);
	g2 = sub_mod_ntt_gpu_2(U, g2, bid);
	g2 = mul_mod_ntt_gpu(g2, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2 + 1];
	U = s_a[operation_id + 2 * t];
	g4 = s_a[operation_id + 3 * t];
	g3 = add_mod_ntt_gpu_2(U, g4, bid);
	g4 = sub_mod_ntt_gpu_2(U, g4, bid);
	g4 = mul_mod_ntt_gpu(g4, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + tid / t];
	U = g1;
	s_a[operation_id] = add_mod_ntt_gpu_2(U, g3, bid);
	g3 = sub_mod_ntt_gpu_2(U, g3, bid);
	s_a[operation_id + 2 * t] = mul_mod_ntt_gpu(g3, S, bid);
	U = g2;
	s_a[operation_id + t] = add_mod_ntt_gpu_2(U, g4, bid);
	g4 = sub_mod_ntt_gpu_2(U, g4, bid);
	s_a[operation_id + 3 * t] = mul_mod_ntt_gpu(g4, S, bid);
	__syncthreads();

	t = t * 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2];
	U = s_a[operation_id];
	g2 = s_a[operation_id + t];
	g1 = add_mod_ntt_gpu_2(U, g2, bid);
	g2 = sub_mod_ntt_gpu_2(U, g2, bid);
	g2 = mul_mod_ntt_gpu(g2, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2 + 1];
	U = s_a[operation_id + 2 * t];
	g4 = s_a[operation_id + 3 * t];
	g3 = add_mod_ntt_gpu_2(U, g4, bid);
	g4 = sub_mod_ntt_gpu_2(U, g4, bid);
	g4 = mul_mod_ntt_gpu(g4, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + tid / t];
	U = g1;
	s_a[operation_id] = add_mod_ntt_gpu_2(U, g3, bid);
	g3 = sub_mod_ntt_gpu_2(U, g3, bid);
	s_a[operation_id + 2 * t] = mul_mod_ntt_gpu(g3, S, bid);
	U = g2;
	s_a[operation_id + t] = add_mod_ntt_gpu_2(U, g4, bid);
	g4 = sub_mod_ntt_gpu_2(U, g4, bid);
	s_a[operation_id + 3 * t] = mul_mod_ntt_gpu(g4, S, bid);
	__syncthreads();

	t = t * 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2];
	U = s_a[operation_id];
	g2 = s_a[operation_id + t];
	g1 = add_mod_ntt_gpu_2(U, g2, bid);
	g2 = sub_mod_ntt_gpu_2(U, g2, bid);
	g2 = mul_mod_ntt_gpu(g2, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2 + 1];
	U = s_a[operation_id + 2 * t];
	g4 = s_a[operation_id + 3 * t];
	g3 = add_mod_ntt_gpu_2(U, g4, bid);
	g4 = sub_mod_ntt_gpu_2(U, g4, bid);
	g4 = mul_mod_ntt_gpu(g4, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + tid / t];
	U = g1;
	s_a[operation_id] = add_mod_ntt_gpu_2(U, g3, bid);
	g3 = sub_mod_ntt_gpu_2(U, g3, bid);
	s_a[operation_id + 2 * t] = mul_mod_ntt_gpu(g3, S, bid);
	U = g2;
	s_a[operation_id + t] = add_mod_ntt_gpu_2(U, g4, bid);
	g4 = sub_mod_ntt_gpu_2(U, g4, bid);
	s_a[operation_id + 3 * t] = mul_mod_ntt_gpu(g4, S, bid);
	__syncthreads();

	t = t * 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2];
	U = s_a[operation_id];
	g2 = s_a[operation_id + t];
	g1 = add_mod_ntt_gpu_2(U, g2, bid);
	g2 = sub_mod_ntt_gpu_2(U, g2, bid);
	g2 = mul_mod_ntt_gpu(g2, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2 + 1];
	U = s_a[operation_id + 2 * t];
	g4 = s_a[operation_id + 3 * t];
	g3 = add_mod_ntt_gpu_2(U, g4, bid);
	g4 = sub_mod_ntt_gpu_2(U, g4, bid);
	g4 = mul_mod_ntt_gpu(g4, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + tid / t];
	U = g1;
	s_a[operation_id] = add_mod_ntt_gpu_2(U, g3, bid);
	g3 = sub_mod_ntt_gpu_2(U, g3, bid);
	s_a[operation_id + 2 * t] = mul_mod_ntt_gpu(g3, S, bid);
	U = g2;
	s_a[operation_id + t] = add_mod_ntt_gpu_2(U, g4, bid);
	g4 = sub_mod_ntt_gpu_2(U, g4, bid);
	s_a[operation_id + 3 * t] = mul_mod_ntt_gpu(g4, S, bid);
	__syncthreads();

	t = t * 4;
	operation_id = (tid / t) * 4 * t + tid % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2];
	U = s_a[operation_id];
	g2 = s_a[operation_id + t];
	g1 = add_mod_ntt_gpu_2(U, g2, bid);
	g2 = sub_mod_ntt_gpu_2(U, g2, bid);
	g2 = mul_mod_ntt_gpu(g2, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + (tid / t) * 2 + 1];
	U = s_a[operation_id + 2 * t];
	g4 = s_a[operation_id + 3 * t];
	g3 = add_mod_ntt_gpu_2(U, g4, bid);
	g4 = sub_mod_ntt_gpu_2(U, g4, bid);
	g4 = mul_mod_ntt_gpu(g4, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + tid / t];
	U = g1;
	s_a[operation_id] = add_mod_ntt_gpu_2(U, g3, bid);
	g3 = sub_mod_ntt_gpu_2(U, g3, bid);
	s_a[operation_id + 2 * t] = mul_mod_ntt_gpu(g3, S, bid);
	U = g2;
	s_a[operation_id + t] = add_mod_ntt_gpu_2(U, g4, bid);
	g4 = sub_mod_ntt_gpu_2(U, g4, bid);
	s_a[operation_id + 3 * t] = mul_mod_ntt_gpu(g4, S, bid);
	__syncthreads();

	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + blockIdx.x*SIFE_N + tid] = mul_mod_ntt_gpu(s_a[tid], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + blockIdx.x*SIFE_N + tid + 1024] = mul_mod_ntt_gpu(s_a[tid + 1024], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + blockIdx.x*SIFE_N + tid + 2048] = mul_mod_ntt_gpu(s_a[tid + 2048], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + blockIdx.x*SIFE_N + tid + 3072] = mul_mod_ntt_gpu(s_a[tid + 3072], SIFE_NTT_NINV_gpu[bid], bid);
}
__global__ void GS_reverse_gpu_1block_3round2(uint32_t* a) {
	int64_t t, S, U, g0, g1, g2, g3, g4, g5, g6, g7;
	uint32_t thread_id, operation_id;
	__shared__ uint32_t s_a[SIFE_N];
	uint32_t bid = blockIdx.x%SIFE_NMODULI;
	uint32_t repeat = blockIdx.y;

	thread_id = threadIdx.x;
	s_a[thread_id] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + bid * SIFE_N];
	s_a[thread_id + 512] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 512 + bid * SIFE_N];
	s_a[thread_id + 1024] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 1024 + bid * SIFE_N];
	s_a[thread_id + 1536] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 1536 + bid * SIFE_N];
	s_a[thread_id + 2048] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 2048 + bid * SIFE_N];
	s_a[thread_id + 2560] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 2560 + bid * SIFE_N];
	s_a[thread_id + 3072] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 3072 + bid * SIFE_N];
	s_a[thread_id + 3584] = a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 3584 + bid * SIFE_N];
	__syncthreads();

	t = 1;
	operation_id = (threadIdx.x / t) * 8 * t + threadIdx.x % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	U = s_a[operation_id];
	g1 = s_a[operation_id + t];
	g0 = add_mod_ntt_gpu_2(U, g1, bid);
	g1 = sub_mod_ntt_gpu_2(U, g1, bid);
	g1 = mul_mod_ntt_gpu(g1, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 1];
	U = s_a[operation_id + t * 2];
	g3 = s_a[operation_id + t * 3];
	g2 = add_mod_ntt_gpu_2(U, g3, bid);
	g3 = sub_mod_ntt_gpu_2(U, g3, bid);
	g3 = mul_mod_ntt_gpu(g3, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 2];
	U = s_a[operation_id + t * 4];
	g5 = s_a[operation_id + t * 5];
	g4 = add_mod_ntt_gpu_2(U, g5, bid);
	g5 = sub_mod_ntt_gpu_2(U, g5, bid);
	g5 = mul_mod_ntt_gpu(g5, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 3];
	U = s_a[operation_id + t * 6];
	g7 = s_a[operation_id + t * 7];
	g6 = add_mod_ntt_gpu_2(U, g7, bid);
	g7 = sub_mod_ntt_gpu_2(U, g7, bid);
	g7 = mul_mod_ntt_gpu(g7, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4];
	U = g0;
	g0 = add_mod_ntt_gpu_2(U, g2, bid);
	g2 = sub_mod_ntt_gpu_2(U, g2, bid);
	g2 = mul_mod_ntt_gpu(g2, S, bid);
	U = g1;
	g1 = add_mod_ntt_gpu_2(U, g3, bid);
	g3 = sub_mod_ntt_gpu_2(U, g3, bid);
	g3 = mul_mod_ntt_gpu(g3, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4 + 1];
	U = g4;
	g4 = add_mod_ntt_gpu_2(U, g6, bid);
	g6 = sub_mod_ntt_gpu_2(U, g6, bid);
	g6 = mul_mod_ntt_gpu(g6, S, bid);
	U = g5;
	g5 = add_mod_ntt_gpu_2(U, g7, bid);
	g7 = sub_mod_ntt_gpu_2(U, g7, bid);
	g7 = mul_mod_ntt_gpu(g7, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 8 + operation_id / t / 8];
	s_a[operation_id] = add_mod_ntt_gpu_2(g0, g4, bid);
	g4 = sub_mod_ntt_gpu_2(g0, g4, bid);
	s_a[operation_id + t * 4] = mul_mod_ntt_gpu(g4, S, bid);
	s_a[operation_id + t] = add_mod_ntt_gpu_2(g1, g5, bid);
	g5 = sub_mod_ntt_gpu_2(g1, g5, bid);
	s_a[operation_id + t * 5] = mul_mod_ntt_gpu(g5, S, bid);
	s_a[operation_id + t * 2] = add_mod_ntt_gpu_2(g2, g6, bid);
	g6 = sub_mod_ntt_gpu_2(g2, g6, bid);
	s_a[operation_id + t * 6] = mul_mod_ntt_gpu(g6, S, bid);
	s_a[operation_id + t * 3] = add_mod_ntt_gpu_2(g3, g7, bid);
	g7 = sub_mod_ntt_gpu_2(g3, g7, bid);
	s_a[operation_id + t * 7] = mul_mod_ntt_gpu(g7, S, bid);
	t = t * 8;
	__syncthreads();

	operation_id = (threadIdx.x / t) * 8 * t + threadIdx.x % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	U = s_a[operation_id];
	g1 = s_a[operation_id + t];
	g0 = add_mod_ntt_gpu_2(U, g1, bid);
	g1 = sub_mod_ntt_gpu_2(U, g1, bid);
	g1 = mul_mod_ntt_gpu(g1, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 1];
	U = s_a[operation_id + t * 2];
	g3 = s_a[operation_id + t * 3];
	g2 = add_mod_ntt_gpu_2(U, g3, bid);
	g3 = sub_mod_ntt_gpu_2(U, g3, bid);
	g3 = mul_mod_ntt_gpu(g3, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 2];
	U = s_a[operation_id + t * 4];
	g5 = s_a[operation_id + t * 5];
	g4 = add_mod_ntt_gpu_2(U, g5, bid);
	g5 = sub_mod_ntt_gpu_2(U, g5, bid);
	g5 = mul_mod_ntt_gpu(g5, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 3];
	U = s_a[operation_id + t * 6];
	g7 = s_a[operation_id + t * 7];
	g6 = add_mod_ntt_gpu_2(U, g7, bid);
	g7 = sub_mod_ntt_gpu_2(U, g7, bid);
	g7 = mul_mod_ntt_gpu(g7, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4];
	U = g0;
	g0 = add_mod_ntt_gpu_2(U, g2, bid);
	g2 = sub_mod_ntt_gpu_2(U, g2, bid);
	g2 = mul_mod_ntt_gpu(g2, S, bid);
	U = g1;
	g1 = add_mod_ntt_gpu_2(U, g3, bid);
	g3 = sub_mod_ntt_gpu_2(U, g3, bid);
	g3 = mul_mod_ntt_gpu(g3, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4 + 1];
	U = g4;
	g4 = add_mod_ntt_gpu_2(U, g6, bid);
	g6 = sub_mod_ntt_gpu_2(U, g6, bid);
	g6 = mul_mod_ntt_gpu(g6, S, bid);
	U = g5;
	g5 = add_mod_ntt_gpu_2(U, g7, bid);
	g7 = sub_mod_ntt_gpu_2(U, g7, bid);
	g7 = mul_mod_ntt_gpu(g7, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 8 + operation_id / t / 8];
	s_a[operation_id] = add_mod_ntt_gpu_2(g0, g4, bid);
	g4 = sub_mod_ntt_gpu_2(g0, g4, bid);
	s_a[operation_id + t * 4] = mul_mod_ntt_gpu(g4, S, bid);
	s_a[operation_id + t] = add_mod_ntt_gpu_2(g1, g5, bid);
	g5 = sub_mod_ntt_gpu_2(g1, g5, bid);
	s_a[operation_id + t * 5] = mul_mod_ntt_gpu(g5, S, bid);
	s_a[operation_id + t * 2] = add_mod_ntt_gpu_2(g2, g6, bid);
	g6 = sub_mod_ntt_gpu_2(g2, g6, bid);
	s_a[operation_id + t * 6] = mul_mod_ntt_gpu(g6, S, bid);
	s_a[operation_id + t * 3] = add_mod_ntt_gpu_2(g3, g7, bid);
	g7 = sub_mod_ntt_gpu_2(g3, g7, bid);
	s_a[operation_id + t * 7] = mul_mod_ntt_gpu(g7, S, bid);
	t = t * 8;
	__syncthreads();

	operation_id = (threadIdx.x / t) * 8 * t + threadIdx.x % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	U = s_a[operation_id];
	g1 = s_a[operation_id + t];
	g0 = add_mod_ntt_gpu_2(U, g1, bid);
	g1 = sub_mod_ntt_gpu_2(U, g1, bid);
	g1 = mul_mod_ntt_gpu(g1, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 1];
	U = s_a[operation_id + t * 2];
	g3 = s_a[operation_id + t * 3];
	g2 = add_mod_ntt_gpu_2(U, g3, bid);
	g3 = sub_mod_ntt_gpu_2(U, g3, bid);
	g3 = mul_mod_ntt_gpu(g3, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 2];
	U = s_a[operation_id + t * 4];
	g5 = s_a[operation_id + t * 5];
	g4 = add_mod_ntt_gpu_2(U, g5, bid);
	g5 = sub_mod_ntt_gpu_2(U, g5, bid);
	g5 = mul_mod_ntt_gpu(g5, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 3];
	U = s_a[operation_id + t * 6];
	g7 = s_a[operation_id + t * 7];
	g6 = add_mod_ntt_gpu_2(U, g7, bid);
	g7 = sub_mod_ntt_gpu_2(U, g7, bid);
	g7 = mul_mod_ntt_gpu(g7, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4];
	U = g0;
	g0 = add_mod_ntt_gpu_2(U, g2, bid);
	g2 = sub_mod_ntt_gpu_2(U, g2, bid);
	g2 = mul_mod_ntt_gpu(g2, S, bid);
	U = g1;
	g1 = add_mod_ntt_gpu_2(U, g3, bid);
	g3 = sub_mod_ntt_gpu_2(U, g3, bid);
	g3 = mul_mod_ntt_gpu(g3, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4 + 1];
	U = g4;
	g4 = add_mod_ntt_gpu_2(U, g6, bid);
	g6 = sub_mod_ntt_gpu_2(U, g6, bid);
	g6 = mul_mod_ntt_gpu(g6, S, bid);
	U = g5;
	g5 = add_mod_ntt_gpu_2(U, g7, bid);
	g7 = sub_mod_ntt_gpu_2(U, g7, bid);
	g7 = mul_mod_ntt_gpu(g7, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 8 + operation_id / t / 8];
	s_a[operation_id] = add_mod_ntt_gpu_2(g0, g4, bid);
	g4 = sub_mod_ntt_gpu_2(g0, g4, bid);
	s_a[operation_id + t * 4] = mul_mod_ntt_gpu(g4, S, bid);
	s_a[operation_id + t] = add_mod_ntt_gpu_2(g1, g5, bid);
	g5 = sub_mod_ntt_gpu_2(g1, g5, bid);
	s_a[operation_id + t * 5] = mul_mod_ntt_gpu(g5, S, bid);
	s_a[operation_id + t * 2] = add_mod_ntt_gpu_2(g2, g6, bid);
	g6 = sub_mod_ntt_gpu_2(g2, g6, bid);
	s_a[operation_id + t * 6] = mul_mod_ntt_gpu(g6, S, bid);
	s_a[operation_id + t * 3] = add_mod_ntt_gpu_2(g3, g7, bid);
	g7 = sub_mod_ntt_gpu_2(g3, g7, bid);
	s_a[operation_id + t * 7] = mul_mod_ntt_gpu(g7, S, bid);
	t = t * 8;
	__syncthreads();

	operation_id = (threadIdx.x / t) * 8 * t + threadIdx.x % t;
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2];
	U = s_a[operation_id];
	g1 = s_a[operation_id + t];
	g0 = add_mod_ntt_gpu_2(U, g1, bid);
	g1 = sub_mod_ntt_gpu_2(U, g1, bid);
	g1 = mul_mod_ntt_gpu(g1, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 1];
	U = s_a[operation_id + t * 2];
	g3 = s_a[operation_id + t * 3];
	g2 = add_mod_ntt_gpu_2(U, g3, bid);
	g3 = sub_mod_ntt_gpu_2(U, g3, bid);
	g3 = mul_mod_ntt_gpu(g3, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 2];
	U = s_a[operation_id + t * 4];
	g5 = s_a[operation_id + t * 5];
	g4 = add_mod_ntt_gpu_2(U, g5, bid);
	g5 = sub_mod_ntt_gpu_2(U, g5, bid);
	g5 = mul_mod_ntt_gpu(g5, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 2 + operation_id / t / 2 + 3];
	U = s_a[operation_id + t * 6];
	g7 = s_a[operation_id + t * 7];
	g6 = add_mod_ntt_gpu_2(U, g7, bid);
	g7 = sub_mod_ntt_gpu_2(U, g7, bid);
	g7 = mul_mod_ntt_gpu(g7, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4];
	U = g0;
	g0 = add_mod_ntt_gpu_2(U, g2, bid);
	g2 = sub_mod_ntt_gpu_2(U, g2, bid);
	g2 = mul_mod_ntt_gpu(g2, S, bid);
	U = g1;
	g1 = add_mod_ntt_gpu_2(U, g3, bid);
	g3 = sub_mod_ntt_gpu_2(U, g3, bid);
	g3 = mul_mod_ntt_gpu(g3, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 4 + operation_id / t / 4 + 1];
	U = g4;
	g4 = add_mod_ntt_gpu_2(U, g6, bid);
	g6 = sub_mod_ntt_gpu_2(U, g6, bid);
	g6 = mul_mod_ntt_gpu(g6, S, bid);
	U = g5;
	g5 = add_mod_ntt_gpu_2(U, g7, bid);
	g7 = sub_mod_ntt_gpu_2(U, g7, bid);
	g7 = mul_mod_ntt_gpu(g7, S, bid);
	S = psi_inv_gpu[bid][SIFE_N / t / 8 + operation_id / t / 8];
	s_a[operation_id] = add_mod_ntt_gpu_2(g0, g4, bid);
	g4 = sub_mod_ntt_gpu_2(g0, g4, bid);
	s_a[operation_id + t * 4] = mul_mod_ntt_gpu(g4, S, bid);
	s_a[operation_id + t] = add_mod_ntt_gpu_2(g1, g5, bid);
	g5 = sub_mod_ntt_gpu_2(g1, g5, bid);
	s_a[operation_id + t * 5] = mul_mod_ntt_gpu(g5, S, bid);
	s_a[operation_id + t * 2] = add_mod_ntt_gpu_2(g2, g6, bid);
	g6 = sub_mod_ntt_gpu_2(g2, g6, bid);
	s_a[operation_id + t * 6] = mul_mod_ntt_gpu(g6, S, bid);
	s_a[operation_id + t * 3] = add_mod_ntt_gpu_2(g3, g7, bid);
	g7 = sub_mod_ntt_gpu_2(g3, g7, bid);
	s_a[operation_id + t * 7] = mul_mod_ntt_gpu(g7, S, bid);
	t = t * 8;
	__syncthreads();

	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + bid * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 512 + bid * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id + 512], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 1024 + bid * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id + 1024], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 1536 + bid * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id + 1536], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 2048 + bid * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id + 2048], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 2560 + bid * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id + 2560], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 3072 + bid * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id + 3072], SIFE_NTT_NINV_gpu[bid], bid);
	a[repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N + thread_id + 3584 + bid * SIFE_N] = mul_mod_ntt_gpu(s_a[thread_id + 3584], SIFE_NTT_NINV_gpu[bid], bid);
}
