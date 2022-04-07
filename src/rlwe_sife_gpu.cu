#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <gmp.h>
#include "params.h"
#include "rlwe_sife.h"
#include "crt.h"
#include "sample.h"
#include "randombytes.h"
#include "ntt.h"
#include "arith_rns.h"
#include "gauss.h"
#include "ntt_gpu.cuh"
#include "sample_gpu.cuh"
#include "AES.cuh"
#include "crt_gpu.cuh"
#include "function.cuh"

#define NUM_BIN 32

// Check only one moduli at a time.
void histogram2(uint32_t data[SIFE_NMODULI][SIFE_N])
{
	uint64_t i, j, k, idx, countpos = 0, countneg = 0;
	uint64_t max = 1, min = 0, sum = 0;
	uint64_t binspos[NUM_BIN], binsneg[NUM_BIN], binsize = 1;
	uint32_t *neg, *pos;

	neg = (uint32_t*) malloc(SIFE_NMODULI*SIFE_N);
	pos = (uint32_t*) malloc(SIFE_NMODULI*SIFE_N);
	// convert positive

		for(j=0; j<1; j++){	// wklee, select only 1 moduli 
			for(k=0; k<SIFE_N; k++){
				if(data[j][k] < SIFE_MOD_Q_I[j]/2)
				{
					pos[countpos] = data[j][k] ;					
						countpos++;
				}		
				else
				{
					neg[countneg] = data[j][k] ;				
						countneg++;				
				}		
			}
		}		
	printf("\n ************Histogram Calculation *************\n");
	printf("Pos elements: %lu Neg elements: %lu \n", countpos, countneg );

// This is for the negative  side
	// find the max
	for(i=0; i<countneg; i++)	{
		if(neg[i] > max)
			max = neg[i];				
	}
	min = max;
	// find the min
	for(i=0; i<countneg; i++)	{
		if(neg[i]< min)
			min = neg[i];
	}
	binsize = (max - min)/NUM_BIN + ((max - min)%NUM_BIN !=0);

	// reset all bins to 0
	for(i=0; i<NUM_BIN; i++) binsneg[i] = 0;
	for(i=0; i<countneg; i++)	{
		idx = (neg[i] - min)/binsize;
		binsneg[idx]++;			
	}		

	printf("\n Negative side ==> max: %lu min: %lu binsize: %lu\n", max, min, binsize);
	for(j=0; j<NUM_BIN; j++) sum += binsneg[j];
	for(j=0; j<NUM_BIN; j++) printf("%lu  ", binsneg[j]);
	printf("\n");

	max = 1;
// This is for the positive side
	// find the max
	for(i=0; i<countpos; i++)	{
		if(pos[i] > max)
			max = pos[i];				
	}
	min = max;
	// find the min
	for(i=0; i<countpos; i++)	{
		if(pos[i]< min)
			min = pos[i];
	}
	binsize = (max - min)/NUM_BIN + ((max - min)%NUM_BIN !=0);
	// reset all bins to 0
	for(i=0; i<NUM_BIN; i++) binspos[i] = 0;
	for(i=0; i<countpos; i++)	{
		idx = (pos[i] - min)/binsize;
		binspos[idx]++;		
	}		
	printf("\n Positive side ==> max: %lu min: %lu binsize: %lu\n", max, min, binsize);	
	for(j=0; j<NUM_BIN; j++) sum += binspos[j];
	for(j=0; j<NUM_BIN; j++) printf("%lu  ", binspos[j]);

	free(neg);
	free(pos);
}

#ifdef PERF
extern "C" void rlwe_sife_setup_gpu(uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], unsigned char *seed2, unsigned char *seed3, float* part2_time)
#else
extern "C" void rlwe_sife_setup_gpu(uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], unsigned char *seed2, unsigned char *seed3)
#endif
{
	uint32_t *d_mpk, *d_msk_ntt, *d_msk, *d_c, *d_ecrt;
	uint32_t *tmp;// *msk_ntt
	uint8_t* dev_rk1, *dev_rk2;
	char* m_EncryptKey1, *m_EncryptKey2;
	int i, j;
#ifdef PERF
	cudaEvent_t start, stop;	
	float elapsed;
	
	cudaEventCreate(&start);	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif	
	cudaMallocHost((void**)&m_EncryptKey1, 16 * 15 *sizeof(char));
	cudaMallocHost((void**)&m_EncryptKey2, 16 * 15 *sizeof(char));
	// cudaMallocHost((void**)&msk_ntt, SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMallocHost((void**)&tmp, SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&d_mpk, SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&d_msk_ntt, SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&d_msk, SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&d_c, (SIFE_L+1)*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&d_ecrt, SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&dev_rk1, 4*60 * sizeof(uint8_t));//AES256
	cudaMalloc((void**)&dev_rk2, 4*60 * sizeof(uint8_t));
	AESPrepareKey(m_EncryptKey1, seed2, 256);
	AESPrepareKey(m_EncryptKey2, seed3, 256);

	//wklee, fix this.
	for(i=0; i<SIFE_NMODULI; i++)
		for(j=0; j<SIFE_N; j++)
			tmp[i*SIFE_N + j] = mpk[SIFE_L][i][j];

	cudaMemcpy(dev_rk1, m_EncryptKey1, 4*60*sizeof(uint8_t),cudaMemcpyHostToDevice);	
	cudaMemcpy(dev_rk2, m_EncryptKey2, 4*60*sizeof(uint8_t),cudaMemcpyHostToDevice);
	cudaMemcpy(d_mpk, tmp, SIFE_NMODULI*SIFE_N*sizeof(uint32_t),cudaMemcpyHostToDevice);		
	
	gaussian_sampler_S1_gpu<<<SIFE_L, THREAD>>>(dev_rk1, d_msk);
	gaussian_sampler_S1_gpu<<<SIFE_L, THREAD>>>(dev_rk2, d_ecrt);
	// wklee: replace this with a kernel to copy data
	cudaMemcpy(d_msk_ntt, d_msk, SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t),cudaMemcpyDeviceToDevice);	
		// Store a in NTT domain
#if SEC_LEVEL==0	
	CT_forward_gpu_1block_1round << <SIFE_NMODULI, 1024 >> > (d_mpk);	
	CT_forward_gpu_1block_1round << <SIFE_L*SIFE_NMODULI, 1024 >> > (d_ecrt);
	CT_forward_gpu_1block_1round << <SIFE_L*SIFE_NMODULI, 1024 >> > (d_msk_ntt);
#elif SEC_LEVEL==1
	CT_forward_gpu_1block_2round << <SIFE_NMODULI, 1024 >> > (d_mpk);	
	CT_forward_gpu_1block_2round << <SIFE_L*SIFE_NMODULI, 1024 >> > (d_ecrt);
	CT_forward_gpu_1block_2round << <SIFE_L*SIFE_NMODULI, 1024 >> > (d_msk_ntt);
#endif
	point_mul_gpu<<<SIFE_L, 1024>>>(d_c, d_mpk, d_msk_ntt);
	point_add_mod_gpu<<<SIFE_L, 1024>>>(d_c, d_c, d_ecrt);
	cudaMemcpy(mpk, d_c, SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(msk, d_msk, SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(tmp, d_mpk, SIFE_NMODULI*SIFE_N*sizeof(uint32_t),cudaMemcpyDeviceToHost);		

	// wklee: replace this with cudaMemcpy
	for(i=0; i<SIFE_NMODULI; i++)
		for(j=0; j<SIFE_N; j++)
			mpk[SIFE_L][i][j] = tmp[i*SIFE_N + j];

#ifdef PERF	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);  
  	cudaEventElapsedTime(&elapsed, start, stop);   
  	//printf("rlwe_sife_setup_gpu part 2: %.4f ms\n", elapsed);     
  	*part2_time += elapsed;
#endif   

	cudaFreeHost(m_EncryptKey1);
	cudaFreeHost(m_EncryptKey2);
	// cudaFreeHost(msk_ntt);
	cudaFreeHost(tmp);
	cudaFree(d_mpk);
	cudaFree(d_msk_ntt);
	cudaFree(d_msk);
	cudaFree(d_c);
	cudaFree(d_ecrt);
	cudaFree(dev_rk1);//AES256
	cudaFree(dev_rk2);
}

#ifdef PERF
extern "C" void rlwe_sife_encrypt_gpu(uint32_t m[SIFE_L], uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], unsigned char *seed2, unsigned char *seed3, float* part2_time)
#else
extern "C" void rlwe_sife_encrypt_gpu(uint32_t m[SIFE_L], uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], unsigned char *seed2, unsigned char *seed3)
#endif
{
	uint32_t *d_mpk, *d_sample, *d_rcrt, *d_c, *d_fcrt, *d_mcrt, *d_m;
	uint32_t *tmp;
	uint8_t* dev_rk1, *dev_rk2;
	char* m_EncryptKey1, *m_EncryptKey2;	
#ifdef PERF
	cudaEvent_t start, stop;	
	float elapsed;
	
	cudaEventCreate(&start);	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif	
	cudaMallocHost((void**)&m_EncryptKey1, 16 * 15 *sizeof(char));
	cudaMallocHost((void**)&m_EncryptKey2, 16 * 15 *sizeof(char));

	cudaMallocHost((void**)&tmp, SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&d_mcrt, SIFE_NMODULI*SIFE_L*sizeof(uint32_t));
	cudaMalloc((void**)&d_mpk, (SIFE_L+1)*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&d_m, SIFE_L*sizeof(uint32_t));
	cudaMalloc((void**)&d_sample, SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&d_c, (SIFE_L+1)*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&d_rcrt, SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&d_fcrt, SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&dev_rk1, 4*60 * sizeof(uint8_t));//AES256
	cudaMalloc((void**)&dev_rk2, 4*60 * sizeof(uint8_t));

	AESPrepareKey(m_EncryptKey1, seed2, 256);
	AESPrepareKey(m_EncryptKey2, seed3, 256);

	cudaMemcpy(dev_rk1, m_EncryptKey1, 4*60*sizeof(uint8_t),cudaMemcpyHostToDevice);	
	cudaMemcpy(dev_rk2, m_EncryptKey2, 4*60*sizeof(uint8_t),cudaMemcpyHostToDevice);
	// cudaMemcpy(d_tmp2, tmp, SIFE_NMODULI*SIFE_N*sizeof(uint32_t),cudaMemcpyHostToDevice);			
	cudaMemcpy(d_m, m, SIFE_L*sizeof(uint32_t),cudaMemcpyHostToDevice);	
	cudaMemcpy(d_mpk, mpk, (SIFE_L+1)*SIFE_NMODULI*SIFE_N*sizeof(uint32_t),cudaMemcpyHostToDevice);		
	
	// CRT and scaled message
	crt_convert_generic_gpu<<<SIFE_NMODULI, SIFE_L>>>(d_m, d_mcrt);
	// needs to be changed. messagges are small no need for reduction
	crt_mxm_gpu<<<SIFE_NMODULI, SIFE_L>>>(d_mcrt);

	// // Sample r, f_0 from D_sigma2
	gaussian_sampler_S2_gpu<<<1, THREAD>>>(dev_rk1, d_rcrt);
	gaussian_sampler_S2_gpu<<<1, THREAD>>>(dev_rk2, d_fcrt);	
	// r in NTT domain
#if SEC_LEVEL==0	
	CT_forward_gpu_1block_1round << <SIFE_NMODULI, 1024 >> > (d_rcrt);	
#elif SEC_LEVEL==1	
	CT_forward_gpu_1block_2round << <SIFE_NMODULI, 1024 >> > (d_rcrt);	
#endif
	point_mul_gpu2<<<SIFE_NMODULI, 1024>>>(d_c+SIFE_L*SIFE_NMODULI*SIFE_N, d_rcrt, d_mpk+SIFE_L*SIFE_NMODULI*SIFE_N);
#if SEC_LEVEL==0	
	GS_reverse_gpu_1block_1round<< <SIFE_NMODULI, 1024 >> > (d_c+SIFE_L*SIFE_NMODULI*SIFE_N);
#elif SEC_LEVEL==1	
	GS_reverse_gpu_1block_2round<< <SIFE_NMODULI, 1024 >> > (d_c+SIFE_L*SIFE_NMODULI*SIFE_N);	
#endif
	point_add_mod_gpu3<<<SIFE_NMODULI, 1024>>>(d_c+SIFE_L*SIFE_NMODULI*SIFE_N, d_c+SIFE_L*SIFE_NMODULI*SIFE_N, d_fcrt);

	// // Sample f_i with i = 1...l from D_sigma3
	// c_i = pk_i * r + f_i + (floor(q/p)m_i)1_R
	gaussian_sampler_S3_gpu<<<SIFE_L, THREAD>>>(dev_rk2, d_sample);
	point_mul_gpu<<<SIFE_L, 1024>>>(d_c, d_rcrt, d_mpk);
#if SEC_LEVEL==0
	GS_reverse_gpu_1block_1round<< <SIFE_L*SIFE_NMODULI, 1024 >> > (d_c);	
#elif SEC_LEVEL==1	
	GS_reverse_gpu_1block_2round<< <SIFE_L*SIFE_NMODULI, 1024 >> > (d_c);	
#endif
	point_add_mod_gpu<<<SIFE_L, 1024>>>(d_c, d_sample, d_c);
	point_add_mod_gpu2<<<SIFE_L, 1024>>>(d_c, d_mcrt);
	
	cudaMemcpy(c, d_c, (SIFE_L+1)*SIFE_NMODULI*SIFE_N*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	// cudaMemcpy(f_crt, d_sample, SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t), cudaMemcpyDeviceToHost);	
	// cudaMemcpy(m_crt, d_mcrt, SIFE_NMODULI*SIFE_L*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	// cudaMemcpy(f_crt, d_fcrt, SIFE_NMODULI*SIFE_N*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	// cudaMemcpy(r_crt, d_rcrt, SIFE_NMODULI*SIFE_N*sizeof(uint32_t), cudaMemcpyDeviceToHost);


	cudaMallocHost((void**)&m_EncryptKey1, 16 * 15 *sizeof(char));
	cudaMallocHost((void**)&m_EncryptKey2, 16 * 15 *sizeof(char));

#ifdef PERF	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);  
  	cudaEventElapsedTime(&elapsed, start, stop);   
  	//printf("rlwe_sife_encrypt part 2: %.4f \n", elapsed);     
  	*part2_time += elapsed; 
#endif   

	cudaFreeHost(m_EncryptKey1);
	cudaFreeHost(m_EncryptKey2);
	cudaFreeHost(tmp);
	cudaFree(d_mcrt);
	cudaFree(d_mpk);
	cudaFree(d_m);
	cudaFree(d_sample);
	cudaFree(d_c);
	cudaFree(d_rcrt);
	cudaFree(d_fcrt);
	cudaFree(dev_rk1);
	cudaFree(dev_rk2);
}

#ifdef PERF
extern "C" void rlwe_sife_keygen_gpu(const uint32_t y[SIFE_L], const uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], uint32_t sk_y[SIFE_NMODULI][SIFE_N], float* part2_time)
#else
extern "C" void rlwe_sife_keygen_gpu(const uint32_t y[SIFE_L], const uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], uint32_t sk_y[SIFE_NMODULI][SIFE_N])
#endif
{
	uint32_t *d_msk, *d_y, *d_ycrt, *d_sky;
	// int i, j;
#ifdef PERF
	cudaEvent_t start, stop;	
	float elapsed;
	
	cudaEventCreate(&start);	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif	
	cudaMalloc((void**)&d_ycrt, SIFE_NMODULI*SIFE_L*sizeof(uint32_t));
	cudaMalloc((void**)&d_sky, SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&d_y, SIFE_L*sizeof(uint32_t));
	cudaMalloc((void**)&d_msk, SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	
	cudaMemcpy(d_y, y, SIFE_L*sizeof(uint32_t),cudaMemcpyHostToDevice);	
	cudaMemcpy(d_msk, msk, SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t),cudaMemcpyHostToDevice);		

	crt_convert_generic_gpu<<<SIFE_NMODULI, SIFE_L>>>(d_y, d_ycrt);
	crt_mul_acc_gpu<<<SIFE_N/1024, 1024>>>(d_msk, d_ycrt, d_sky);

	cudaMemcpy(sk_y, d_sky, SIFE_NMODULI*SIFE_N*sizeof(uint32_t), cudaMemcpyDeviceToHost);

#ifdef PERF		
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);  
  	cudaEventElapsedTime(&elapsed, start, stop);   
  	//printf("rlwe_sife_keygen_gpu Latency %.4f (ms)\n", elapsed);
  	*part2_time += elapsed; 
#endif   

	cudaFree(d_ycrt);
	cudaFree(d_sky);
	cudaFree(d_y);
	cudaFree(d_msk);
}

#ifdef PERF
extern "C" void rlwe_sife_decrypt_gpu(uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], const uint32_t y[SIFE_L], uint32_t sk_y[SIFE_NMODULI][SIFE_N], uint32_t d_y[SIFE_NMODULI][SIFE_N], float* part2_time)
#else
extern "C" void rlwe_sife_decrypt_gpu(uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], const uint32_t y[SIFE_L], uint32_t sk_y[SIFE_NMODULI][SIFE_N], uint32_t d_y[SIFE_NMODULI][SIFE_N])
#endif
{
	uint32_t *d_c, *d_yarray, *d_ycrt, *dev_dy, *d_sky, *d_c0sy;
#ifdef PERF
	cudaEvent_t start, stop;	
	float elapsed;
	
	cudaEventCreate(&start);	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif	
	cudaMalloc((void**)&d_ycrt, SIFE_NMODULI*SIFE_L*sizeof(uint32_t));
	cudaMalloc((void**)&dev_dy, SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&d_sky, SIFE_NMODULI*SIFE_N*sizeof(uint32_t));	
	cudaMalloc((void**)&d_c0sy, SIFE_NMODULI*SIFE_N*sizeof(uint32_t));	
	cudaMalloc((void**)&d_yarray, SIFE_L*sizeof(uint32_t));
	cudaMalloc((void**)&d_c, (SIFE_L+1)*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	
	cudaMemcpy(d_yarray, y, SIFE_L*sizeof(uint32_t),cudaMemcpyHostToDevice);	
	cudaMemcpy(d_c, c, (SIFE_L+1)*SIFE_NMODULI*SIFE_N*sizeof(uint32_t),cudaMemcpyHostToDevice);	
	cudaMemcpy(d_sky, sk_y, SIFE_NMODULI*SIFE_N*sizeof(uint32_t),cudaMemcpyHostToDevice);		

	crt_convert_generic_gpu<<<SIFE_NMODULI, SIFE_L>>>(d_yarray, d_ycrt);
	crt_mul_acc_gpu<<<SIFE_N/1024, 1024>>>(d_c, d_ycrt, dev_dy);

#if SEC_LEVEL==0
	CT_forward_gpu_1block_1round << <SIFE_NMODULI, 1024 >> > (d_sky);	
	CT_forward_gpu_1block_1round << <SIFE_NMODULI, 1024 >> > (d_c+SIFE_L*SIFE_NMODULI*SIFE_N);	
	point_mul_gpu2<<<SIFE_NMODULI, 1024>>>(d_c0sy, d_sky, d_c+SIFE_L*SIFE_NMODULI*SIFE_N);
	GS_reverse_gpu_1block_1round<< <SIFE_NMODULI, 1024 >> > (d_c0sy);		
#elif SEC_LEVEL==1	
	#if CT_TEST==112
	CT_forward_gpu_1block_2round << <SIFE_NMODULI, 1024 >> > (d_sky);	
	CT_forward_gpu_1block_2round << <SIFE_NMODULI, 1024 >> > (d_c+SIFE_L*SIFE_NMODULI*SIFE_N);	
	#elif CT_TEST==113
	CT_forward_gpu_1block_3round << <SIFE_NMODULI, 512 >> > (d_sky);	
	CT_forward_gpu_1block_3round << <SIFE_NMODULI, 512 >> > (d_c+SIFE_L*SIFE_NMODULI*SIFE_N);	
	#elif CT_TEST==721
	CT_forward_gpu_64block_2kernel_1round_1_batch << <SIFE_NMODULI*64, 32 >> > (d_sky);	
	CT_forward_gpu_64block_2kernel_1round_2_batch << <SIFE_NMODULI*64, 32 >> > (d_sky);	
	CT_forward_gpu_64block_2kernel_1round_1_batch << <SIFE_NMODULI*64, 32 >> > (d_c+SIFE_L*SIFE_NMODULI*SIFE_N);	
	CT_forward_gpu_64block_2kernel_1round_2_batch << <SIFE_NMODULI*64, 32 >> > (d_c+SIFE_L*SIFE_NMODULI*SIFE_N);	
	#endif	

	point_mul_gpu2<<<SIFE_NMODULI, 1024>>>(d_c0sy, d_sky, d_c+SIFE_L*SIFE_NMODULI*SIFE_N);

	#if GS_TEST==112
	GS_reverse_gpu_1block_2round<< <SIFE_NMODULI, 1024 >> > (d_c0sy);	
	#elif GS_TEST==113
	GS_reverse_gpu_1block_3round<< <SIFE_NMODULI, 512 >> > (d_c0sy);	
	#elif GS_TEST==721
	GS_reverse_gpu_64block_2kernel_1round_1_batch << <SIFE_NMODULI*64, 32 >> > (d_c0sy);
	GS_reverse_gpu_64block_2kernel_1round_2_batch << <SIFE_NMODULI*64, 32 >> > (d_c0sy);
	#endif	
#endif	

	poly_sub_mod_gpu<< <SIFE_NMODULI, 1024 >> >(dev_dy, d_c0sy, dev_dy);

	// cudaMemcpy(y_crt, d_ycrt, SIFE_NMODULI*SIFE_L*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(d_y, dev_dy, SIFE_NMODULI*SIFE_N*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	// cudaMemcpy(c0sy, d_c0sy, SIFE_NMODULI*SIFE_N*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	// cudaMemcpy(c, d_c, (SIFE_L+1)*SIFE_NMODULI*SIFE_L*sizeof(uint32_t), cudaMemcpyDeviceToHost);

#ifdef PERF		
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);  
  	cudaEventElapsedTime(&elapsed, start, stop);   
  	//printf("rlwe_sife_decrypt_gmp part 2 %.4f ms \n", elapsed);     
  	*part2_time += elapsed; 
#endif   

	cudaFree(d_ycrt);
	cudaFree(dev_dy);
	cudaFree(d_sky);	
	cudaFree(d_c0sy);	
	cudaFree(d_yarray);
	cudaFree(d_c);
}




#ifdef PERF
extern "C" void rlwe_sife_encrypt_gpu2(uint32_t* m, uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t* c, unsigned char *seed2, unsigned char *seed3, int repeat, float* part2_time)
#else
extern "C" void rlwe_sife_encrypt_gpu2(uint32_t* m, uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t* c, unsigned char *seed2, unsigned char *seed3, int repeat)
#endif
{
	uint32_t *d_mpk, *d_rcrt, *d_c, *d_fcrt, *d_mcrt, *d_m;
	uint8_t* dev_rk1, *dev_rk2;
	char* m_EncryptKey1, *m_EncryptKey2;
#ifdef PERF
	cudaEvent_t start, stop;	
	float elapsed;
	
	cudaEventCreate(&start);	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif	
	cudaMallocHost((void**)&m_EncryptKey1, repeat*4*60 *sizeof(char));
	cudaMallocHost((void**)&m_EncryptKey2, repeat*4*60 *sizeof(char));

	cudaMalloc((void**)&d_mcrt, repeat * SIFE_NMODULI*SIFE_L*sizeof(uint32_t));
	cudaMalloc((void**)&d_mpk, (SIFE_L+1)*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&d_m, repeat * SIFE_L*sizeof(uint32_t));
	cudaMalloc((void**)&d_c, repeat * (SIFE_L+1)*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&d_rcrt, repeat * SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&d_fcrt, repeat * SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&dev_rk1, repeat * 4*60 * sizeof(uint8_t));//AES256
	cudaMalloc((void**)&dev_rk2, repeat * 4*60 * sizeof(uint8_t));

    for(int i=0; i<repeat; i++)
    {
		AESPrepareKey(m_EncryptKey1 + i*4*60, seed2, 256);
		AESPrepareKey(m_EncryptKey2 + i*4*60, seed3, 256);
	}

	cudaMemcpy(dev_rk1, m_EncryptKey1, repeat*4*60*sizeof(uint8_t),cudaMemcpyHostToDevice);	
	cudaMemcpy(dev_rk2, m_EncryptKey1, repeat*4*60*sizeof(uint8_t),cudaMemcpyHostToDevice);	
	cudaMemcpy(d_mpk, mpk, (SIFE_L+1)*SIFE_NMODULI*SIFE_N*sizeof(uint32_t),cudaMemcpyHostToDevice);
	cudaMemcpy(d_m, m, repeat*SIFE_L*sizeof(uint32_t),cudaMemcpyHostToDevice);	

	dim3 grid1(1, repeat);
	dim3 grid2(SIFE_NMODULI, repeat);
	dim3 grid3(SIFE_L*SIFE_NMODULI, repeat);
	dim3 grid4(SIFE_L, repeat);

	// // Sample r, f_0 from D_sigma2
	gaussian_sampler_S2_gpu<<<grid1, THREAD>>>(dev_rk1, d_rcrt);
	gaussian_sampler_S2_gpu<<<grid1, THREAD>>>(dev_rk2, d_fcrt);	

	// CRT and scaled message
	crt_convert_generic_gpu<<<grid2, SIFE_L>>>(d_m, d_mcrt);
	// needs to be changed. messagges are small no need for reduction
	crt_mxm_gpu<<<grid2, SIFE_L>>>(d_mcrt);

	// r in NTT domain
#if CT_TEST==112
	CT_forward_gpu_1block_2round << <grid2, 1024 >> > (d_rcrt);	
#elif CT_TEST==113
	CT_forward_gpu_1block_3round << <grid2, 512 >> > (d_rcrt);	
#elif CT_TEST==721
	dim3 grid5(SIFE_NMODULI*64, repeat);
	CT_forward_gpu_64block_2kernel_1round_1_batch << <grid5, 32 >> > (d_rcrt);	
	CT_forward_gpu_64block_2kernel_1round_2_batch << <grid5, 32 >> > (d_rcrt);	
#endif	
	point_mul_gpu2_2<<<grid2, 1024>>>(d_c+SIFE_L*SIFE_NMODULI*SIFE_N, d_rcrt, d_mpk+SIFE_L*SIFE_NMODULI*SIFE_N);
#if GS_TEST==112
	GS_reverse_gpu_1block_2round2<< <grid2, 1024 >> > (d_c+SIFE_L*SIFE_NMODULI*SIFE_N);	
#elif GS_TEST==113
	GS_reverse_gpu_1block_3round2<< <grid2, 512 >> > (d_c+SIFE_L*SIFE_NMODULI*SIFE_N);	
#elif GS_TEST==721
	dim3 grid6(SIFE_NMODULI*64, repeat);
	GS_reverse_gpu_64block_2kernel_1round_1_batch2 << <grid6, 32 >> > (d_c+SIFE_L*SIFE_NMODULI*SIFE_N);
	GS_reverse_gpu_64block_2kernel_1round_2_batch2 << <grid6, 32 >> > (d_c+SIFE_L*SIFE_NMODULI*SIFE_N);
#endif	
	point_add_mod_gpu3_2<<<grid2, 1024>>>(d_c+SIFE_L*SIFE_NMODULI*SIFE_N, d_c+SIFE_L*SIFE_NMODULI*SIFE_N, d_fcrt);

	// // Sample f_i with i = 1...l from D_sigma3
	// c_i = pk_i * r + f_i + (floor(q/p)m_i)1_R
	point_mul_gpu_2<<<grid4, 1024>>>(d_c, d_rcrt, d_mpk);
#if GS_TEST==112
	GS_reverse_gpu_1block_2round2<< <grid3, 1024 >> > (d_c);	
#elif GS_TEST==113
	GS_reverse_gpu_1block_3round2<< <grid3, 512 >> > (d_c);	
#elif GS_TEST==721
	dim3 grid7(SIFE_L*SIFE_NMODULI*64, repeat);
	GS_reverse_gpu_64block_2kernel_1round_1_batch2 << <grid7, 32 >> > (d_c);
	GS_reverse_gpu_64block_2kernel_1round_2_batch2 << <grid7, 32 >> > (d_c);
#endif	
	gaussian_sampler_S3_gpu2<<<grid4, 1024>>>(dev_rk2, d_c);
	point_add_mod_gpu2_2<<<grid4, 1024>>>(d_c, d_mcrt);
	
	cudaMemcpy(c, d_c, repeat*(SIFE_L+1)*SIFE_NMODULI*SIFE_N*sizeof(uint32_t), cudaMemcpyDeviceToHost);

#ifdef PERF	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);  
  	cudaEventElapsedTime(&elapsed, start, stop);   
  	//printf("rlwe_sife_encrypt part 2: %.4f \n", elapsed);     
  	*part2_time += elapsed; 
#endif   

	cudaFreeHost(m_EncryptKey1);
	cudaFreeHost(m_EncryptKey2);
	cudaFree(d_mcrt);
	cudaFree(d_mpk);
	cudaFree(d_m);
	cudaFree(d_c);
	cudaFree(d_rcrt);
	cudaFree(d_fcrt);
	cudaFree(dev_rk1);
	cudaFree(dev_rk2);
}

#ifdef PERF
extern "C" void rlwe_sife_keygen_gui2(const uint32_t* y, const uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], uint32_t* sk_y, int repeat, float* part2_time)
#else
extern "C" void rlwe_sife_keygen_gui2(const uint32_t* y, const uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], uint32_t* sk_y, int repeat)
#endif
{
	uint32_t *d_msk, *d_y, *d_sky;
	// int i, j;
#ifdef PERF
	cudaEvent_t start, stop;	
	float elapsed;
	
	cudaEventCreate(&start);	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif	
	cudaMalloc((void**)&d_y, repeat * SIFE_L*sizeof(uint32_t));
	cudaMalloc((void**)&d_msk, SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&d_sky, repeat * SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	
	cudaMemcpy(d_y, y, repeat * SIFE_L*sizeof(uint32_t),cudaMemcpyHostToDevice);	
	cudaMemcpy(d_msk, msk, SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t),cudaMemcpyHostToDevice);		

	dim3 grid1(SIFE_NMODULI, repeat);
	keygen_gpu<<<grid1, 1024>>>(d_y, d_msk, d_sky);

	cudaMemcpy(sk_y, d_sky, repeat * SIFE_NMODULI*SIFE_N*sizeof(uint32_t), cudaMemcpyDeviceToHost);

#ifdef PERF		
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);  
  	cudaEventElapsedTime(&elapsed, start, stop);   
  	//printf("rlwe_sife_keygen_gpu Latency %.4f (ms)\n", elapsed);
  	*part2_time += elapsed; 
#endif   

	cudaFree(d_sky);
	cudaFree(d_y);
	cudaFree(d_msk);
}

#ifdef PERF	
extern "C" void rlwe_sife_decrypt_gmp_gui2(uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], const uint32_t* y, uint32_t* sk_y, uint32_t* d_y, int repeat, float* part2_time)  
#else
extern "C" void rlwe_sife_decrypt_gmp_gui2(uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], const uint32_t* y, uint32_t* sk_y, uint32_t* d_y, int repeat)
#endif   
{
	uint32_t *d_c, *d_yarray, *dev_dy, *d_sky;
#ifdef PERF
	cudaEvent_t start, stop;	
	float elapsed;
	
	cudaEventCreate(&start);	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif	
	cudaMalloc((void**)&d_c, (SIFE_L+1)*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&d_yarray, repeat * SIFE_L*sizeof(uint32_t));
	cudaMalloc((void**)&d_sky, repeat * SIFE_NMODULI*SIFE_N*sizeof(uint32_t));	
	cudaMalloc((void**)&dev_dy, repeat * SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	
	cudaMemcpy(d_c, c, (SIFE_L+1)*SIFE_NMODULI*SIFE_N*sizeof(uint32_t),cudaMemcpyHostToDevice);	
	cudaMemcpy(d_yarray, y, repeat * SIFE_L*sizeof(uint32_t),cudaMemcpyHostToDevice);	
	cudaMemcpy(d_sky, sk_y, repeat * SIFE_NMODULI*SIFE_N*sizeof(uint32_t),cudaMemcpyHostToDevice);		

	dim3 grid1(SIFE_NMODULI, repeat);
	decryption_gpu<<<grid1, 1024>>>(d_yarray, d_c, d_sky, dev_dy);

	cudaMemcpy(d_y, dev_dy, repeat * SIFE_NMODULI*SIFE_N*sizeof(uint32_t), cudaMemcpyDeviceToHost);

#ifdef PERF		
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);  
  	cudaEventElapsedTime(&elapsed, start, stop);   
  	//printf("rlwe_sife_decrypt_gmp part 2 %.4f ms \n", elapsed);    
  	*part2_time += elapsed; 
#endif   

	cudaFree(d_yarray);
	cudaFree(d_c);
	cudaFree(d_sky);	
	cudaFree(dev_dy);
}


#ifdef PERF
extern "C" void rlwe_sife_keygen_gui3(const uint32_t* y, const uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], uint32_t* sk_y, int repeat, float* part2_time)
#else
extern "C" void rlwe_sife_keygen_gui3(const uint32_t* y, const uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], uint32_t* sk_y, int repeat)
#endif
{
	uint32_t *d_msk, *d_y, *d_ycrt, *d_sky;
	// int i, j;
#ifdef PERF
	cudaEvent_t start, stop;	
	float elapsed;
	
	cudaEventCreate(&start);	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif	
	cudaMalloc((void**)&d_ycrt, repeat * SIFE_NMODULI*SIFE_L*sizeof(uint32_t));
	cudaMalloc((void**)&d_y, repeat * SIFE_L*sizeof(uint32_t));
	cudaMalloc((void**)&d_msk, SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&d_sky, repeat * SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	
	cudaMemcpy(d_y, y, repeat * SIFE_L*sizeof(uint32_t),cudaMemcpyHostToDevice);	
	cudaMemcpy(d_msk, msk, SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t),cudaMemcpyHostToDevice);		

	dim3 grid1(SIFE_NMODULI, repeat);
	dim3 grid2(SIFE_N/1024, repeat);

	crt_convert_generic_gpu<<<grid1, SIFE_L>>>(d_y, d_ycrt);
	crt_mul_acc_gpu2<<<grid2, 1024>>>(d_msk, d_ycrt, d_sky);

	cudaMemcpy(sk_y, d_sky, repeat * SIFE_NMODULI*SIFE_N*sizeof(uint32_t), cudaMemcpyDeviceToHost);

#ifdef PERF		
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);  
  	cudaEventElapsedTime(&elapsed, start, stop);   
  	//printf("rlwe_sife_keygen_gpu Latency %.4f (ms)\n", elapsed);
  	*part2_time += elapsed; 
#endif   

	cudaFree(d_ycrt);
	cudaFree(d_sky);
	cudaFree(d_y);
	cudaFree(d_msk);
}

#ifdef PERF	
extern "C" void rlwe_sife_decrypt_gmp_gui3(uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], const uint32_t* y, uint32_t* sk_y, uint32_t* d_y, int repeat, float* part2_time)  
#else
extern "C" void rlwe_sife_decrypt_gmp_gui3(uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], const uint32_t* y, uint32_t* sk_y, uint32_t* d_y, int repeat)
#endif   
{
	uint32_t *d_c, *d_yarray, *d_ycrt, *dev_dy, *d_sky, *d_c0sy;
#ifdef PERF
	cudaEvent_t start, stop;	
	float elapsed;
	
	cudaEventCreate(&start);	cudaEventCreate(&stop);
	cudaEventRecord(start);
#endif	
	cudaMalloc((void**)&d_ycrt, repeat*SIFE_NMODULI*SIFE_L*sizeof(uint32_t));
	cudaMalloc((void**)&d_c0sy, repeat*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));	
	cudaMalloc((void**)&d_c, (SIFE_L+1)*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&d_yarray, repeat*SIFE_L*sizeof(uint32_t));
	cudaMalloc((void**)&d_sky, repeat*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));	
	cudaMalloc((void**)&dev_dy, repeat*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	
	cudaMemcpy(d_c, c, (SIFE_L+1)*SIFE_NMODULI*SIFE_N*sizeof(uint32_t),cudaMemcpyHostToDevice);	
	cudaMemcpy(d_yarray, y, repeat*SIFE_L*sizeof(uint32_t),cudaMemcpyHostToDevice);	
	cudaMemcpy(d_sky, sk_y, repeat*SIFE_NMODULI*SIFE_N*sizeof(uint32_t),cudaMemcpyHostToDevice);		

	dim3 grid1(SIFE_NMODULI, repeat);
	dim3 grid2(SIFE_N/1024, repeat);
	dim3 grid3(SIFE_NMODULI*64, repeat);

	crt_convert_generic_gpu<<<grid1, SIFE_L>>>(d_yarray, d_ycrt);
	crt_mul_acc_gpu2<<<grid2, 1024>>>(d_c, d_ycrt, dev_dy);

	#if CT_TEST==112
	CT_forward_gpu_1block_2round << <grid1, 1024 >> > (d_sky);	
	CT_forward_gpu_1block_2round << <SIFE_NMODULI, 1024 >> > (d_c+SIFE_L*SIFE_NMODULI*SIFE_N);	
	#elif CT_TEST==113
	CT_forward_gpu_1block_3round << <grid1, 512 >> > (d_sky);	
	CT_forward_gpu_1block_3round << <SIFE_NMODULI, 512 >> > (d_c+SIFE_L*SIFE_NMODULI*SIFE_N);	
	#elif CT_TEST==721
	CT_forward_gpu_64block_2kernel_1round_1_batch << <grid3, 32 >> > (d_sky);	
	CT_forward_gpu_64block_2kernel_1round_2_batch << <grid3, 32 >> > (d_sky);	
	CT_forward_gpu_64block_2kernel_1round_1_batch << <SIFE_NMODULI*64, 32 >> > (d_c+SIFE_L*SIFE_NMODULI*SIFE_N);	
	CT_forward_gpu_64block_2kernel_1round_2_batch << <SIFE_NMODULI*64, 32 >> > (d_c+SIFE_L*SIFE_NMODULI*SIFE_N);	
	#endif	

	point_mul_gpu2_3<<<grid1, 1024>>>(d_c0sy, d_sky, d_c+SIFE_L*SIFE_NMODULI*SIFE_N);

	#if GS_TEST==112
	GS_reverse_gpu_1block_2round<< <grid1, 1024 >> > (d_c0sy);	
	#elif GS_TEST==113
	GS_reverse_gpu_1block_3round<< <grid1, 512 >> > (d_c0sy);	
	#elif GS_TEST==721
	GS_reverse_gpu_64block_2kernel_1round_1_batch3 << <grid3, 32 >> > (d_c0sy);
	GS_reverse_gpu_64block_2kernel_1round_2_batch3 << <grid3, 32 >> > (d_c0sy);
	#endif	
	poly_sub_mod_gpu2<< <grid1, 1024 >> >(dev_dy, d_c0sy, dev_dy);

	cudaMemcpy(d_y, dev_dy, repeat * SIFE_NMODULI*SIFE_N*sizeof(uint32_t), cudaMemcpyDeviceToHost);

#ifdef PERF		
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);  
  	cudaEventElapsedTime(&elapsed, start, stop);   
  	//printf("rlwe_sife_decrypt_gmp part 2 %.4f ms \n", elapsed);    
  	*part2_time += elapsed; 
#endif   

	cudaFree(d_ycrt);
	cudaFree(d_c0sy);	
	cudaFree(d_yarray);
	cudaFree(d_c);
	cudaFree(d_sky);	
	cudaFree(dev_dy);
}
