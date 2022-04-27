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
#include "aes256ctr.h"

long long cpucycles(void)
{
    unsigned long long result;
    asm volatile(".byte 15;.byte 49;shlq $32,%%rdx;orq %%rdx,%%rax"
        : "=a" (result) ::  "%rdx");
    return result;
}

#ifdef PERF	
extern void rlwe_sife_setup_gpu(uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], unsigned char *seed2, unsigned char *seed3, float* part2_time);
extern void rlwe_sife_encrypt_gpu(uint32_t* m, uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t* c, unsigned char *seed2, unsigned char *seed3, int repeat, float* part2_time);
#else
extern void rlwe_sife_setup_gpu(uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], unsigned char *seed2, unsigned char *seed3);
extern void rlwe_sife_encrypt_gpu(uint32_t* m, uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t* c, unsigned char *seed2, unsigned char *seed3, int repeat);
#endif   

#ifdef PERF	
void rlwe_sife_setup(uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], double* time)  
#else
void rlwe_sife_setup(uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N]) 
#endif   
{
	int i, j;
	uint32_t e_crt[SIFE_NMODULI][SIFE_N];
	uint32_t msk_ntt[SIFE_N];
	aes256ctr_ctx state_secret, state_error;
	unsigned char seed[32];

#ifdef PERF	
    uint64_t CLOCK1, CLOCK2;
    CLOCK1=cpucycles();
#endif   
	randombytes(seed, 32);

	sample_polya(seed, mpk[SIFE_L]);

	randombytes(seed, 32);
	aes256ctr_init(&state_error, seed, 0);

	// Store a in NTT domain
	for (i = 0; i < SIFE_NMODULI; ++i) {
		CT_forward(mpk[SIFE_L][i], i);
	}

	// Sample s_i and e_i with i = 1...l from D_sigma1
	// pk_i = a * s_i + e_i
	for (i = 0; i < SIFE_L; ++i) {
#ifdef AVX2
		gaussian_sampler_S1_ori(&state_secret, msk[i], SIFE_N);
		gaussian_sampler_S1_ori(&state_error, e_crt, SIFE_N);
#else
		gaussian_sampler_S1(&state_secret, msk[i], SIFE_N);
		gaussian_sampler_S1(&state_error, e_crt, SIFE_N);
#endif
		for (j = 0; j < SIFE_NMODULI; ++j) {
			// Store pk_i in NTT domain but not s_i
			memcpy(msk_ntt, msk[i][j], SIFE_N*sizeof(uint32_t));
			CT_forward(msk_ntt, j);
			CT_forward(e_crt[j], j);
			point_mul(mpk[SIFE_L][j], msk_ntt, mpk[i][j], j);
			poly_add_mod(mpk[i][j], e_crt[j], mpk[i][j], j);
		}
	}
#ifdef PERF	
	CLOCK2=cpucycles();
    *time += (double)CLOCK2 - CLOCK1;
#endif
}

#ifdef PERF	
void rlwe_sife_setup_gui(uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], double* part1_time, float* part2_time)  
#else
void rlwe_sife_setup_gui(uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N]) 
#endif   
{
	// int i, j, k;
	unsigned char seed1[32]={0}, seed2[32]={0}, seed3[32]={0};
#ifdef PERF	
    uint64_t CLOCK1, CLOCK2;
    CLOCK1=cpucycles();
#endif   
	randombytes(seed1, 32);
	sample_polya(seed1, mpk[SIFE_L]);
	// Sample s_i and e_i with i = 1...l from D_sigma1
	// pk_i = a * s_i + e_i	
	randombytes(seed2, 32);
	randombytes(seed3, 32);
#ifdef PERF	
	CLOCK2=cpucycles();
    *part1_time += (double)CLOCK2 - CLOCK1;
	rlwe_sife_setup_gpu(mpk, msk, seed2, seed3, part2_time);
#else
	rlwe_sife_setup_gpu(mpk, msk, seed2, seed3);
#endif    
}

#ifdef PERF	
void rlwe_sife_encrypt(uint32_t m[SIFE_L], uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], double* time)  
#else
void rlwe_sife_encrypt(uint32_t m[SIFE_L], uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N]) //add const keywords
#endif   
{
	int i, j, k;
	uint32_t r_crt[SIFE_NMODULI][SIFE_N], f_crt[SIFE_NMODULI][SIFE_N];
	uint32_t m_crt[SIFE_NMODULI][SIFE_L];
	aes256ctr_ctx state_s2, state_s3;
	unsigned char seed[32];

#ifdef PERF	
    uint64_t CLOCK1, CLOCK2;
    CLOCK1=cpucycles();
#endif   

	// CRT and scaled message
	crt_convert_generic(m, m_crt, SIFE_L); // needs to be changed. messagges are small no need for reduction

	uint64_t mxm;
	for (i = 0; i < SIFE_L; ++i) {
		for (j = 0; j < SIFE_NMODULI; ++j) {
			mxm = (uint64_t)m_crt[j][i] * SIFE_SCALE_M_MOD_Q_I[j];
			m_crt[j][i] = mod_prime(mxm, j);
		}
	}

	randombytes(seed, 32);
	aes256ctr_init(&state_s2, seed, 0);

	// Sample r, f_0 from D_sigma2

#ifdef AVX2
	gaussian_sampler_S2_ori(&state_s2, r_crt, SIFE_N);
	gaussian_sampler_S2_ori(&state_s2, f_crt, SIFE_N);
#else
	gaussian_sampler_S2(&state_s2, r_crt, SIFE_N);
	gaussian_sampler_S2(&state_s2, f_crt, SIFE_N);
#endif

	// r in NTT domain
	for (i = 0; i < SIFE_NMODULI; ++i) {
		CT_forward(r_crt[i], i);
	}

	for (i = 0; i < SIFE_NMODULI; ++i) {
		point_mul(mpk[SIFE_L][i], r_crt[i], c[SIFE_L][i], i);
		GS_reverse(c[SIFE_L][i], i);
		poly_add_mod(c[SIFE_L][i], f_crt[i], c[SIFE_L][i], i);
	}

	// Sample f_i with i = 1...l from D_sigma3
	// c_i = pk_i * r + f_i + (floor(q/p)m_i)1_R
	for (i = 0; i < SIFE_L; ++i) {
		gaussian_sampler_S3(&state_s3, f_crt, SIFE_N);
		
		for (j = 0; j < SIFE_NMODULI; ++j) {
			point_mul(mpk[i][j], r_crt[j], c[i][j], j);
			GS_reverse(c[i][j], j);
			poly_add_mod(c[i][j], f_crt[j], c[i][j], j);
			for (k = 0; k < SIFE_N; ++k) {
				c[i][j][k] = mod_prime((c[i][j][k] + m_crt[j][i]), j);
			}
		}
	}
#ifdef PERF	
	CLOCK2=cpucycles();
    *time += (double)CLOCK2 - CLOCK1;
#endif   
}

#ifdef PERF	
void rlwe_sife_encrypt_gui(uint32_t* m, uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t* c, int repeat, double* part1_time, float* part2_time)  
#else
void rlwe_sife_encrypt_gui(uint32_t* m, uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t* c, int repeat) //add const keywords
#endif   
{
	int i, j, k, l;
	unsigned char seed[32];

#ifdef PERF	
	uint64_t CLOCK1, CLOCK2;
    CLOCK1=cpucycles();
#endif   
	randombytes(seed, 32);
#ifdef PERF	
	CLOCK2=cpucycles();
	*part1_time += CLOCK2 - CLOCK1;
	//printf("rlwe_sife_encrypt part 1: %lu cycles\n", CLOCK2-CLOCK1);
#endif   
	// aes256ctr_init(&state_s2, seed, 0);

	// Sample r, f_0 from D_sigma2
#ifdef PERF
	rlwe_sife_encrypt_gpu(m, mpk, c, seed, seed, repeat, part2_time);
	//printf("rlwe_sife_encrypt part 2: %.4f \n", *part2_time);     
#else
	rlwe_sife_encrypt_gpu(m, mpk, c, seed, seed, repeat);
#endif   
	// printf("\n c: \n"); for (j = 0; j <SIFE_L+1; j++)  for (k = 0; k < SIFE_NMODULI; k++) for (l = 0; l < SIFE_N; l++) printf("%u ", c[j][k][l]);		
}

#ifdef PERF	
void rlwe_sife_keygen(const uint32_t y[SIFE_L], const uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], uint32_t sk_y[SIFE_NMODULI][SIFE_N], double* time)  
#else
void rlwe_sife_keygen(const uint32_t y[SIFE_L], const uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], uint32_t sk_y[SIFE_NMODULI][SIFE_N])
#endif   
{
	int i, j, k;
	uint64_t mac;
	uint32_t y_crt[SIFE_NMODULI][SIFE_L];

#ifdef PERF	
	uint64_t CLOCK1, CLOCK2;
	CLOCK1=cpucycles();
#endif   

	crt_convert_generic(y, y_crt, SIFE_L);
	memset(sk_y, 0, SIFE_NMODULI*SIFE_N*sizeof(uint32_t));

	for (i = 0; i < SIFE_L; ++i) {
		for (j = 0; j < SIFE_NMODULI; ++j) {
			for (k = 0; k < SIFE_N; ++k) {
				mac = (uint64_t)y_crt[j][i]*msk[i][j][k];
				mac = mac + sk_y[j][k];
				sk_y[j][k] = mod_prime(mac, j);
			}
		}
	}
#ifdef PERF	
	CLOCK2=cpucycles();
	*time += CLOCK2 - CLOCK1;
#endif   
}

#ifdef PERF	
void rlwe_sife_decrypt_gmp(uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], const uint32_t y[SIFE_L], uint32_t sk_y[SIFE_NMODULI][SIFE_N], mpz_t dy[SIFE_N], double* time)  
#else
void rlwe_sife_decrypt_gmp(uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], const uint32_t y[SIFE_L], uint32_t sk_y[SIFE_NMODULI][SIFE_N], mpz_t dy[SIFE_N])
#endif   
{
	int i, j, k;
	uint64_t mac;
	uint32_t c0sy[SIFE_NMODULI][SIFE_N];
	uint32_t d_y[SIFE_NMODULI][SIFE_N] = {0};
	uint32_t y_crt[SIFE_NMODULI][SIFE_L];

#ifdef PERF	
	uint64_t CLOCK1, CLOCK2;
	CLOCK1=cpucycles();
#endif   

	crt_convert_generic(y, y_crt, SIFE_L);
	for (i = 0; i < SIFE_L; ++i) {
		for (j = 0; j < SIFE_NMODULI; ++j) {
			for (k = 0; k < SIFE_N; ++k) {
				mac = (uint64_t)y_crt[j][i]*c[i][j][k];
				mac = mac + d_y[j][k];
				d_y[j][k] = mod_prime(mac, j);
			}
		}
	}

	for (i = 0; i < SIFE_NMODULI; ++i) {
		poly_mul_ntt(c[SIFE_L][i], sk_y[i], c0sy[i], i);
		poly_sub_mod(d_y[i], c0sy[i], d_y[i], i);
	}

	//crt_reverse(dy, d_y);
	crt_reverse_gmp(dy, d_y);
#ifdef PERF	
	CLOCK2=cpucycles();
	*time += CLOCK2 - CLOCK1;
#endif   
}

void round_extract_gmp(mpz_t a[SIFE_N])
{
	int i;

	mpz_t quotient, rem, a_i, SIFE_Q_gmp, SIFE_Q_gmp_by2, SIFE_P_gmp;
	mpz_init(quotient);
	mpz_init(rem);
	mpz_init(a_i);
	mpz_init(SIFE_Q_gmp);
	mpz_init(SIFE_P_gmp);
	mpz_init(SIFE_Q_gmp_by2);

	if(mpz_set_str(SIFE_Q_gmp, SIFE_Q_str, 10)!=0){
		printf("--ERROR unable to set Q to gmp--\n");
		return;
	}

	if(mpz_set_str(SIFE_P_gmp, SIFE_P_str, 10)!=0){
		printf("--ERROR unable to set P to gmp--\n");
		return;
	}

	mpz_fdiv_q_ui(SIFE_Q_gmp_by2, SIFE_Q_gmp, 2);
	//gmp_printf("d[0]: %Zd\n", a[0]);
	for (i = 0; i < SIFE_N; ++i) {
		//mpz_set_ui(a_i, a[i]);
		mpz_set(a_i, a[i]);
		//mpz_mul_ui(a_i, a_i, SIFE_P);
		mpz_mul(a_i, a_i, SIFE_P_gmp);
		//mpz_fdiv_qr_ui(quotient, rem, a_i, SIFE_Q);
		mpz_fdiv_qr(quotient, rem, a_i, SIFE_Q_gmp);
		//if( mpz_cmp_ui(rem, (SIFE_Q_gmp >> 1)) > 0 ) {
		if( mpz_cmp(rem, SIFE_Q_gmp_by2 ) > 0 ) {
			mpz_add_ui(quotient, quotient, 1);
		}
		//a[i] = mpz_get_ui(quotient);
		mpz_set(a[i], quotient);
	}
	mpz_clear(quotient);
	mpz_clear(rem);
	mpz_clear(a_i);
	mpz_clear(SIFE_Q_gmp);
	mpz_clear(SIFE_P_gmp);
	mpz_clear(SIFE_Q_gmp_by2);
}

double round_extract_gmp2(uint32_t d_y[SIFE_NMODULI][SIFE_N])
{
	int i;
	mpz_t a[SIFE_N];
	double result;

	mpz_t quotient, rem, a_i, SIFE_Q_gmp, SIFE_Q_gmp_by2, SIFE_P_gmp;
	mpz_init(quotient);
	mpz_init(rem);
	mpz_init(a_i);
	mpz_init(SIFE_Q_gmp);
	mpz_init(SIFE_P_gmp);
	mpz_init(SIFE_Q_gmp_by2);

    for(int h=0;h<SIFE_N;h++){
        mpz_init(a[h]);
    }
	crt_reverse_gmp(a, d_y);

	if(mpz_set_str(SIFE_Q_gmp, SIFE_Q_str, 10)!=0){
		printf("--ERROR unable to set Q to gmp--\n");
		return 0;
	}

	if(mpz_set_str(SIFE_P_gmp, SIFE_P_str, 10)!=0){
		printf("--ERROR unable to set P to gmp--\n");
		return 0;
	}

	mpz_fdiv_q_ui(SIFE_Q_gmp_by2, SIFE_Q_gmp, 2);
	//gmp_printf("d[0]: %Zd\n", a[0]);
	for (i = 0; i < SIFE_N; ++i) {
		//mpz_set_ui(a_i, a[i]);
		mpz_set(a_i, a[i]);
		//mpz_mul_ui(a_i, a_i, SIFE_P);
		mpz_mul(a_i, a_i, SIFE_P_gmp);
		//mpz_fdiv_qr_ui(quotient, rem, a_i, SIFE_Q);
		mpz_fdiv_qr(quotient, rem, a_i, SIFE_Q_gmp);
		//if( mpz_cmp_ui(rem, (SIFE_Q_gmp >> 1)) > 0 ) {
		if( mpz_cmp(rem, SIFE_Q_gmp_by2 ) > 0 ) {
			mpz_add_ui(quotient, quotient, 1);
		}
		//a[i] = mpz_get_ui(quotient);
		mpz_set(a[i], quotient);
	}

	result = mpz_get_d(a[0]);
    for(int h=0;h<SIFE_N;h++){
        mpz_clear(a[h]);
    }

	mpz_clear(quotient);
	mpz_clear(rem);
	mpz_clear(a_i);
	mpz_clear(SIFE_Q_gmp);
	mpz_clear(SIFE_P_gmp);
	mpz_clear(SIFE_Q_gmp_by2);

	return result;
}