#ifndef GAUSS_H
#define GAUSS_H

#include "aes256ctr.h"
#include "params.h"

static void print128_num(__m128i var) 
{
    int64_t v64val[2];
    memcpy(v64val, &var, sizeof(v64val));
    printf("%.16lx %.16lx\n", v64val[1], v64val[0]);
}

static void print128_num_d(__m128d var) 
{
    double v64val[2];
    memcpy(v64val, &var, sizeof(v64val));
    printf("%.16f %.16f \n", v64val[1], v64val[0]);
}

static void print256_num(__m256i var) 
{
    int64_t v64val[4];
    memcpy(v64val, &var, sizeof(v64val));
    printf("%.16lx %.16lx %.16lx %.16lx\n", v64val[3], v64val[2], v64val[1], v64val[0]);
}

static void print256_num_i(__m256i var) 
{
    int64_t v64val[4];
    memcpy(v64val, &var, sizeof(v64val));
    printf("%.16ld %.16ld %.16ld %.16ld\n", v64val[3], v64val[2], v64val[1], v64val[0]);
}

static void print256_num_d(__m256d var) 
{
    double v64val[4];
    memcpy(v64val, &var, sizeof(v64val));
    printf("%.16f %.16f %.16f %.16f\n", v64val[3], v64val[2], v64val[1], v64val[0]);
}
/*
void gaussian_sampler_S1(unsigned char *seed, int64_t sample[SIFE_NMODULI][SIFE_N], uint32_t slen);

void gaussian_sampler_S2(unsigned char *seed, int64_t sample[SIFE_NMODULI][SIFE_N], uint32_t slen);

void gaussian_sampler_S3(unsigned char *seed, int64_t sample[SIFE_NMODULI][SIFE_N], uint32_t slen);
*/

void gaussian_sampler_S1(aes256ctr_ctx *state, uint32_t sample[SIFE_NMODULI][SIFE_N], uint32_t slen);
void gaussian_sampler_S1_ori(aes256ctr_ctx *state, uint32_t sample[SIFE_NMODULI][SIFE_N], uint32_t slen);

void gaussian_sampler_S2(aes256ctr_ctx *state, uint32_t sample[SIFE_NMODULI][SIFE_N], uint32_t slen);
void gaussian_sampler_S2_ori(aes256ctr_ctx *state, uint32_t sample[SIFE_NMODULI][SIFE_N], uint32_t slen);

void gaussian_sampler_S3(aes256ctr_ctx *state, uint32_t sample[SIFE_NMODULI][SIFE_N], uint32_t slen);

#endif
