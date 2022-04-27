#ifndef RLWE_SIFE_H
#define RLWE_SIFE_H

#include <gmp.h>
#include "params.h"

long long cpucycles(void);

#ifdef PERF	
void rlwe_sife_setup (uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], double* time);
void rlwe_sife_setup_gui(uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], double* part1_time, float* part2_time);
void rlwe_sife_encrypt (uint32_t m[SIFE_L], uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], double* time);
void rlwe_sife_encrypt_gui(uint32_t* m, uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t* c, int repeat, double* part1_time, float* part2_time);
void rlwe_sife_keygen (const uint32_t y[SIFE_L], const uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], uint32_t sk_y[SIFE_NMODULI][SIFE_N], double* time);
void rlwe_sife_decrypt_gmp(uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], const uint32_t y[SIFE_L], uint32_t sk_y[SIFE_NMODULI][SIFE_N], mpz_t dy[SIFE_N], double* time);
#else
void rlwe_sife_setup (uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N]);
void rlwe_sife_setup_gui(uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N]);
void rlwe_sife_encrypt (uint32_t m[SIFE_L], uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N]);
void rlwe_sife_encrypt_gui(uint32_t* m, uint32_t mpk[SIFE_L+1][SIFE_NMODULI][SIFE_N], uint32_t* c, int repeat);
void rlwe_sife_keygen (const uint32_t y[SIFE_L], const uint32_t msk[SIFE_L][SIFE_NMODULI][SIFE_N], uint32_t sk_y[SIFE_NMODULI][SIFE_N]);
void rlwe_sife_decrypt_gmp(uint32_t c[SIFE_L+1][SIFE_NMODULI][SIFE_N], const uint32_t y[SIFE_L], uint32_t sk_y[SIFE_NMODULI][SIFE_N], mpz_t dy[SIFE_N]);
#endif   

void round_extract_gmp(mpz_t a[SIFE_N]);
double round_extract_gmp2(uint32_t d_y[SIFE_NMODULI][SIFE_N]);

#endif
