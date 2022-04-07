
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "params.h"
#include "AES.cuh"
#include "sampler.cuh"
#include "arith_rns.cuh"
#include "randombytes.h"
// #include "params_gpu.cuh"
#include <stdio.h>
// #include <math.h>


#define NUM_BIN 32

// Check only one moduli at a time.
void histogram1(uint32_t data[SIFE_L][SIFE_NMODULI][SIFE_N])
{
	uint64_t i, j, k, idx, countpos = 0, countneg = 0;
	uint64_t max = 1, min = 0, sum = 0;
	uint64_t binspos[NUM_BIN], binsneg[NUM_BIN], binsize = 1;
	uint32_t *neg, *pos;

	neg = (uint32_t*) malloc(SIFE_L*SIFE_NMODULI*SIFE_N);
	pos = (uint32_t*) malloc(SIFE_L*SIFE_NMODULI*SIFE_N);
	// convert positive
	for(i=0; i<SIFE_L; i++)	{
		for(j=0; j<1; j++){	// wklee, select only 1 moduli 
			for(k=0; k<SIFE_N; k++){
				if(data[i][j][k] < SIFE_MOD_Q_I[j]/2)
				{
					pos[countpos] = data[i][j][k] ;					
						countpos++;
				}		
				else
				{
					neg[countneg] = data[i][j][k] ;				
						countneg++;				
				}		
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
// Check only one moduli at a time.
void histogram3(uint32_t *data)
{
	uint64_t i, j, k, idx, countpos = 0, countneg = 0;
	uint64_t max = 1, min = 0, sum = 0;
	uint64_t binspos[NUM_BIN], binsneg[NUM_BIN], binsize = 1;
	uint32_t *neg, *pos;

	neg = (uint32_t*) malloc(SIFE_L*SIFE_NMODULI*SIFE_N);
	pos = (uint32_t*) malloc(SIFE_L*SIFE_NMODULI*SIFE_N);

	// convert positive
	for(i=0; i<SIFE_L; i++)	{
		for(j=2; j<3; j++){	// wklee, select only 1 moduli 
			for(k=0; k<SIFE_N; k++){
				if(data[i*SIFE_NMODULI*SIFE_N + j*SIFE_N + k] < SIFE_MOD_Q_I[j]/2)
				{
					pos[countpos] = data[i*SIFE_NMODULI*SIFE_N + j*SIFE_N + k] ;					
						countpos++;
				}		
				else
				{
					neg[countneg] = data[i*SIFE_NMODULI*SIFE_N + j*SIFE_N + k] ;					
						countneg++;				
				}		
			}
		}		
	}
	printf("\n **************Histogram Calculation *************\n");
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

	// This is for the positive side
	// find the max
	max = 1;
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
	printf("\n");

	free(neg);
	free(pos);
}


double std_dev_f(uint64_t *data, uint64_t size) {
    double sum = 0.0, mean, SD = 0, SD2 = 0;
    double max = 1, min = 0;
    uint64_t i;
    max = 1.0;
		for(i=0; i<size; i++)	{
			if(data[i] > max)
				max = data[i];				
		}
		min = max;
		// find the min
		for(i=0; i<size; i++)	{
			if(data[i]< min)
				min = data[i];
		}
    for (i = 0; i < size; ++i) {
        sum += data[i];
    }
    mean = sum / size;
    for (i = 0; i < size; ++i) {
        SD2 += pow(data[i] - mean, 2);
    }
    SD = sqrt(SD2 / size);
    printf("\n max: %.0f \t min: %.0f \tmean: %.4f SD: %.4f\n", max, min, mean, SD);
    return SD;

}


__global__ void gaussian_sampler_S1_gpu(uint8_t *rk, uint32_t *sample)
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
	uint8_t r[384] = {0};
	uint32_t rep = 0;// Count no. of AES samp. done in each thread
#if SEC_LEVEL==2
	uint32_t mod4;
#endif
	while (j < LEN_THREAD)// not adjustable now, one loop 3 samples.
	{
		do
		{			
			if (i == 8)
			{
				for(l=0; l<4; l++) vx64[l] = 0;				
				aes256ctr_squeezeblocks_gpu (r, AES_ROUNDS, (uint32_t*)rk, rep);
				uniform_sampler_S1_gpu(r + 2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE), v64_y1, v64_y2);

				r1 = r;
				cdt_sampler_gpu(r1, vx64);				    			
				for(l=0; l<4; l++) vx64[l] = (uint32_t) vx64[l] * BINARY_SAMPLER_K_S1;
				for(l=0; l<4; l++) z[l] = vx64[l] + v64_y1[l];	
				for(l=0; l<4; l++) vb_in_64[l] = z[l] + vx64[l];
				for(l=0; l<4; l++) vb_in_64[l] = (uint32_t) vb_in_64[l] * v64_y1[l];
				bernoulli_sampler_S1_gpu(b, vb_in_64, r1 + BASE_TABLE_SIZE);

				r1 = r + BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE;
				for(l=0; l<4; l++) vx64[l] = 0;
				cdt_sampler_gpu(r1, vx64);	
				for(l=0; l<4; l++) vx64[l] = (uint32_t) vx64[l] * BINARY_SAMPLER_K_S1;
				for(l=0; l<4; l++) z[l+4] = vx64[l] + v64_y2[l];
				for(l=0; l<4; l++) vb_in_64[l] = z[l+4] + vx64[l];
				for(l=0; l<4; l++) vb_in_64[l] = (uint32_t) vb_in_64[l] * v64_y2[l];
				bernoulli_sampler_S1_gpu(b + 4, vb_in_64, r1 + BASE_TABLE_SIZE);
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

		sample[j + 0*SIFE_N + bid*SIFE_N*SIFE_NMODULI + tid*LEN_THREAD]=(1-k)*mod1+k*mod_prime_gpu(SIFE_MOD_Q_I_GPU[0]-mod1, 0);
		sample[j + 1*SIFE_N + bid*SIFE_N*SIFE_NMODULI + tid*LEN_THREAD]=(1-k)*mod2+k*mod_prime_gpu(SIFE_MOD_Q_I_GPU[1]-mod2, 1);
		sample[j + 2*SIFE_N + bid*SIFE_N*SIFE_NMODULI + tid*LEN_THREAD]=(1-k)*mod3+k*mod_prime_gpu(SIFE_MOD_Q_I_GPU[2]-mod3, 2);
		#if SEC_LEVEL==2
			mod4=mod_prime_gpu(mod, 3);	
			sample[j + 3*SIFE_N + bid*SIFE_N*SIFE_NMODULI + tid*LEN_THREAD]=(1-k)*mod4+k*mod_prime_gpu(SIFE_MOD_Q_I_GPU[3]-mod4, 3);
		#endif
		j++;
	}
}

__global__ void gaussian_sampler_S2_gpu(uint8_t *rk, uint32_t *sample)
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
#if SEC_LEVEL==2
	uint32_t mod4;
#endif
	while (j < LEN_THREAD)// not adjustable now, one loop 3 samples.
	{
		do
		{			
			if (i == 8)
			{
				for(l=0; l<4; l++) vx64[l] = 0;				
				aes256ctr_squeezeblocks_gpu (r, AES_ROUNDS, (uint32_t*)rk + repeat*4*60, rep);
				uniform_sampler_S2_gpu(r + 2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE), v64_y1, v64_y2);

				r1 = r;
				cdt_sampler_gpu(r1, vx64);				    			
				for(l=0; l<4; l++) vx64[l] = (uint32_t) vx64[l] * BINARY_SAMPLER_K_S2;
				for(l=0; l<4; l++) z[l] = vx64[l] + v64_y1[l];	
				for(l=0; l<4; l++) vb_in_64[l] = z[l] + vx64[l];
				for(l=0; l<4; l++) vb_in_64[l] = (uint32_t) vb_in_64[l] * v64_y1[l];
				bernoulli_sampler_S2_gpu(b, vb_in_64, r1 + BASE_TABLE_SIZE);

				r1 = r + BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE;
				for(l=0; l<4; l++) vx64[l] = 0;
				cdt_sampler_gpu(r1, vx64);	
				for(l=0; l<4; l++) vx64[l] = (uint32_t) vx64[l] * BINARY_SAMPLER_K_S2;
				for(l=0; l<4; l++) z[l+4] = vx64[l] + v64_y2[l];
				for(l=0; l<4; l++) vb_in_64[l] = z[l+4] + vx64[l];
				for(l=0; l<4; l++) vb_in_64[l] = (uint32_t) vb_in_64[l] * v64_y2[l];
				bernoulli_sampler_S2_gpu(b + 4, vb_in_64, r1 + BASE_TABLE_SIZE);
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

		sample[repeat*SIFE_NMODULI*SIFE_N+ j + 0*SIFE_N + bid*SIFE_N*SIFE_NMODULI+ tid*LEN_THREAD]=(1-k)*mod1+k*mod_prime_gpu(SIFE_MOD_Q_I_GPU[0]-mod1, 0);
		sample[repeat*SIFE_NMODULI*SIFE_N+ j + 1*SIFE_N + bid*SIFE_N*SIFE_NMODULI+ tid*LEN_THREAD]=(1-k)*mod2+k*mod_prime_gpu(SIFE_MOD_Q_I_GPU[1]-mod2, 1);
		sample[repeat*SIFE_NMODULI*SIFE_N+ j + 2*SIFE_N + bid*SIFE_N*SIFE_NMODULI+ tid*LEN_THREAD]=(1-k)*mod3+k*mod_prime_gpu(SIFE_MOD_Q_I_GPU[2]-mod3, 2);
		#if SEC_LEVEL==2
			mod4=mod_prime_gpu(mod, 3);	
			sample[repeat*SIFE_NMODULI*SIFE_N+ j + 3*SIFE_N + bid*SIFE_N*SIFE_NMODULI+ tid*LEN_THREAD]=(1-k)*mod4+k*mod_prime_gpu(SIFE_MOD_Q_I_GPU[3]-mod4, 3);
		#endif
		j++;
	}
}

__global__ void gaussian_sampler_S3_gpu(uint8_t *rk, uint32_t *sample)
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
	uint8_t r[384] = {0};
	uint32_t rep = 0;// Count no. of AES samp. done in each thread
#if SEC_LEVEL==2
	uint32_t mod4;
#endif
	while (j < LEN_THREAD)// not adjustable now, one loop 3 samples.
	{
		do
		{			
			if (i == 8)
			{
				for(l=0; l<4; l++) vx64[l] = 0;				
				aes256ctr_squeezeblocks_gpu (r, AES_ROUNDS, (uint32_t*)rk, rep);
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

		sample[j + 0*SIFE_N + bid*SIFE_N*SIFE_NMODULI + tid * LEN_THREAD]=(1-k)*mod1+k*mod_prime_gpu(SIFE_MOD_Q_I_GPU[0]-mod1, 0);
		sample[j + 1*SIFE_N + bid*SIFE_N*SIFE_NMODULI+ tid * LEN_THREAD]=(1-k)*mod2+k*mod_prime_gpu(SIFE_MOD_Q_I_GPU[1]-mod2, 1);
		sample[j + 2*SIFE_N + bid*SIFE_N*SIFE_NMODULI+ tid * LEN_THREAD]=(1-k)*mod3+k*mod_prime_gpu(SIFE_MOD_Q_I_GPU[2]-mod3, 2);
		#if SEC_LEVEL==2
			mod4=mod_prime_gpu(mod, 3);	
			sample[j + 3*SIFE_N + bid*SIFE_N*SIFE_NMODULI+ tid * LEN_THREAD]=(1-k)*mod4+k*mod_prime_gpu(SIFE_MOD_Q_I_GPU[3]-mod4, 3);
		#endif
		j++;
	}
}

extern "C" int gaussian_S1_gpu(unsigned char *seed, uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N])
{	
	// cudaDeviceProp deviceProp;
	// cudaGetDeviceProperties(&deviceProp, 0);
	// printf("\nGPU Compute Capability = [%d.%d], clock: %d asynCopy: %d MapHost: %d SM: %d\n",
	// 	deviceProp.major, deviceProp.minor, deviceProp.clockRate, deviceProp.asyncEngineCount, deviceProp.canMapHostMemory, deviceProp.multiProcessorCount);
	cudaEvent_t start, stop;
	

	// int i, j, k, l;
	uint8_t* dev_rk;
									
	char* m_EncryptKey = (char*)malloc(16 * 15 * sizeof(char));	// Expanded Keys
	uint32_t *d_msk;
	uint64_t *d_clock_c, *clock_c;

	cudaEventCreate(&start);	cudaEventCreate(&stop);
	// cudaMallocHost((void**)&msk, 4*SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMallocHost((void**)&clock_c, SIFE_L*THREAD*sizeof(uint64_t));

	cudaMalloc((void**)&dev_rk, 4*60 * sizeof(uint8_t));	//AES256
	cudaMalloc((void**)&d_msk, 4*SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t));
	cudaMalloc((void**)&d_clock_c, SIFE_L*THREAD*sizeof(uint64_t));

	for (int i = 0; i < 15 * 16; i++)	m_EncryptKey[i] = 0;		
	AESPrepareKey(m_EncryptKey, seed, 256);
#ifdef PERF
	cudaEventRecord(start);
#endif	
	cudaMemcpy(dev_rk, m_EncryptKey, 4*60*sizeof(uint8_t),cudaMemcpyHostToDevice);
	gaussian_sampler_S1_gpu<<<SIFE_L, THREAD>>>(dev_rk, d_msk);
#ifdef PERF	
	float elapsed;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);  
  cudaEventElapsedTime(&elapsed, start, stop);   
  printf("Latency (ms)\n" );
  printf("%.4f \n", elapsed);     
#endif    
  cudaMemcpy(msk, d_msk, SIFE_L*SIFE_NMODULI*SIFE_N*sizeof(uint32_t), cudaMemcpyDeviceToHost);
  // cudaMemcpy(clock_c, d_clock_c, SIFE_L*THREAD*sizeof(uint64_t), cudaMemcpyDeviceToHost);
	// printf("%.4f \n", std_dev_f(msk, 4));
	// histogram1(msk);
	// histogram2(clock_c, SIFE_L*THREAD);
	// std_dev_f(clock_c, SIFE_L*THREAD);
	// printf("\n clock: \n");  for (j = 0; j < SIFE_L*THREAD; j++) printf("%lu\n", clock_c[j]); 
	// printf("\n msk: \n"); for (j = 0; j < SIFE_L; j++)  for (k = 0; k < SIFE_NMODULI; k++) for (l = 0; l < SIFE_N; l++) printf("%u ", msk[j*SIFE_NMODULI*SIFE_N + k*SIFE_N + l]);

	// printf("\n msk: \n"); for (j = 0; j < 1; j++)  for (k = 0; k < SIFE_NMODULI; k++) for (l = 0; l < SIFE_N; l++) if(msk[j*SIFE_NMODULI*SIFE_N + k*SIFE_N + l]!=0)printf("%u %u\n", j*SIFE_NMODULI*SIFE_N + k*SIFE_N + l, msk[j*SIFE_NMODULI*SIFE_N + k*SIFE_N + l]);

	// printf("\n msk: \n"); for (j = 0; j < SIFE_L; j++)  for (k = 0; k < SIFE_NMODULI; k++) for (l = 0; l < SIFE_N; l++) printf("%u %u \n", j*SIFE_NMODULI*SIFE_N + k*SIFE_N + l, msk[j*SIFE_NMODULI*SIFE_N + k*SIFE_N + l]);
	// cudaDeviceReset must be called before exiting in order for profiling and tracing tools such as Nsight and Visual Profiler to show complete traces.
	// cudaDeviceReset();

	free(m_EncryptKey);
	// cudaFreeHost(msk);
	cudaFreeHost(clock_c);
	cudaFree(dev_rk);
	cudaFree(d_msk);
	cudaFree(d_clock_c);

	return 0;
}
