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

	//Comment, fix this.
	for(i=0; i<SIFE_NMODULI; i++)
		for(j=0; j<SIFE_N; j++)
			tmp[i*SIFE_N + j] = mpk[SIFE_L][i][j];

	cudaMemcpy(dev_rk1, m_EncryptKey1, 4*60*sizeof(uint8_t),cudaMemcpyHostToDevice);	
	cudaMemcpy(dev_rk2, m_EncryptKey2, 4*60*sizeof(uint8_t),cudaMemcpyHostToDevice);
	cudaMemcpy(d_mpk, tmp, SIFE_NMODULI*SIFE_N*sizeof(uint32_t),cudaMemcpyHostToDevice);		
	
	gaussian_sampler_S1_gpu<<<SIFE_L, THREAD>>>(dev_rk1, d_msk);
	gaussian_sampler_S1_gpu<<<SIFE_L, THREAD>>>(dev_rk2, d_ecrt);
	// Comment: replace this with a kernel to copy data
	cudaMemcpy(d_msk_ntt, d_msk, SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t),cudaMemcpyDeviceToDevice);	

	// Store a in NTT domain
	CT_forward_gpu_1block_3round << <SIFE_NMODULI, 512 >> > (d_mpk);	
	CT_forward_gpu_1block_3round << <SIFE_L*SIFE_NMODULI, 512 >> > (d_ecrt);
	CT_forward_gpu_1block_3round << <SIFE_L*SIFE_NMODULI, 512 >> > (d_msk_ntt);

	point_mul_gpu<<<SIFE_L, 1024>>>(d_c, d_mpk, d_msk_ntt);
	point_add_mod_gpu<<<SIFE_L, 1024>>>(d_c, d_c, d_ecrt);
	cudaMemcpy(mpk, d_c, SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(msk, d_msk, SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(tmp, d_mpk, SIFE_NMODULI*SIFE_N*sizeof(uint32_t),cudaMemcpyDeviceToHost);		

	// Comment: replace this with cudaMemcpy
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
extern "C" void rlwe_sife_encrypt_gpu(uint32_t* m, uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t* c, unsigned char *seed2, unsigned char *seed3, int repeat, float* part2_time)
#else
extern "C" void rlwe_sife_encrypt_gpu(uint32_t* m, uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t* c, unsigned char *seed2, unsigned char *seed3, int repeat)
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
	CT_forward_gpu_1block_3round << <grid2, 512 >> > (d_rcrt);	

	point_mul_gpu3<<<grid2, 1024>>>(d_c+SIFE_L*SIFE_NMODULI*SIFE_N, d_rcrt, d_mpk+SIFE_L*SIFE_NMODULI*SIFE_N);
	GS_reverse_gpu_1block_3round<< <grid2, 512 >> > (d_c+SIFE_L*SIFE_NMODULI*SIFE_N);	
	point_add_mod_gpu3<<<grid2, 1024>>>(d_c+SIFE_L*SIFE_NMODULI*SIFE_N, d_c+SIFE_L*SIFE_NMODULI*SIFE_N, d_fcrt);

	// // Sample f_i with i = 1...l from D_sigma3
	// c_i = pk_i * r + f_i + (floor(q/p)m_i)1_R
	point_mul_gpu2<<<grid4, 1024>>>(d_c, d_rcrt, d_mpk);
	GS_reverse_gpu_1block_3round<< <grid3, 512 >> > (d_c);	
	gaussian_sampler_S3_gpu<<<grid4, 1024>>>(dev_rk2, d_c);
	point_add_mod_gpu2<<<grid4, 1024>>>(d_c, d_mcrt);
	
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
extern "C" void rlwe_sife_keygen_gpu(const uint32_t* y, const uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], uint32_t* sk_y, int repeat, float* part2_time)
#else
extern "C" void rlwe_sife_keygen_gpu(const uint32_t* y, const uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], uint32_t* sk_y, int repeat)
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
extern "C" void rlwe_sife_decrypt_gmp_gpu(uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], const uint32_t* y, uint32_t* sk_y, uint32_t* d_y, int repeat, float* part2_time)  
#else
extern "C" void rlwe_sife_decrypt_gmp_gpu(uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], const uint32_t* y, uint32_t* sk_y, uint32_t* d_y, int repeat)
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