#include <stdint.h>
#include <stdio.h>
#include <x86intrin.h>
#include "params.h"
#include "arith_rns.h"
#include "randombytes.h"
#include "aes256ctr.h"
#include "gauss.h"
#include <math.h>

/*ceil(DESIRED_SIGMA/sigma_0) sigma_0=sqrt( 1/(2*log(2) ) )*/
#if SEC_LEVEL==0

//sigma_1 : 33 sigma_2 : 64880641 sigma_3 : 129761280
/*
#define BINARY_SAMPLER_K_S1 39		//sigma_1
#define BINARY_SAMPLER_K_S2 76391117LL	//sigma_2
#define BINARY_SAMPLER_K_S3 152782232LL	//sigma_3
*/

//sigma_1 : 33 sigma_2 : 59473921 sigma_3 : 118947840

#define BINARY_SAMPLER_K_S1 39		//sigma_1
#define BINARY_SAMPLER_K_S2 70025191LL	//sigma_2
#define BINARY_SAMPLER_K_S3 140050379LL	//sigma_3


#elif SEC_LEVEL==1

#define BINARY_SAMPLER_K_S1 266		//sigma_1
#define BINARY_SAMPLER_K_S2 304214978LL	//sigma_2
#define BINARY_SAMPLER_K_S3 608429953LL	//sigma_3

#elif SEC_LEVEL==2

#define BINARY_SAMPLER_K_S1 2413
#define BINARY_SAMPLER_K_S2 2029291141LL
#define BINARY_SAMPLER_K_S3 2232220255LL

#endif

#define CDT_ENTRY_SIZE 16
#define CDT_LOW_MASK 0x7fffffffffffffff
#define CDT_LENGTH 9 /* [0..tau*sigma]=[0..9] */

#define BERNOULLI_ENTRY_SIZE 9 /* 72bit randomness */

/* -1/k^2 */
#define BINARY_SAMPLER_K_2_INV_S1 (-1.0/(BINARY_SAMPLER_K_S1 * BINARY_SAMPLER_K_S1))
#define BINARY_SAMPLER_K_2_INV_S2 (-1.0/(BINARY_SAMPLER_K_S2 * BINARY_SAMPLER_K_S2))
#define BINARY_SAMPLER_K_2_INV_S3 (-1.0/(BINARY_SAMPLER_K_S3 * BINARY_SAMPLER_K_S3))


#define EXP_MANTISSA_PRECISION 52
#define EXP_MANTISSA_MASK ((1LL << EXP_MANTISSA_PRECISION) - 1)
#define R_MANTISSA_PRECISION (EXP_MANTISSA_PRECISION + 1)
#define R_MANTISSA_MASK ((1LL << R_MANTISSA_PRECISION) - 1)
#define R_EXPONENT_L (8 * BERNOULLI_ENTRY_SIZE - R_MANTISSA_PRECISION)

#define DOUBLE_ONE (1023LL << 52)

#define UNIFORM_SIZE 4
#define UNIFORM_REJ 20
#define BARRETT_BITSHIFT (UNIFORM_SIZE * 8)

#define BARRETT_FACTOR_S1 ((1LL << BARRETT_BITSHIFT) / BINARY_SAMPLER_K_S1)
#define BARRETT_FACTOR_S2 ((1LL << BARRETT_BITSHIFT) / BINARY_SAMPLER_K_S2)
#define BARRETT_FACTOR_S3 ((1LL << BARRETT_BITSHIFT) / BINARY_SAMPLER_K_S3)

#define UNIFORM_Q_S1 (BINARY_SAMPLER_K_S1 * BARRETT_FACTOR_S1)
#define UNIFORM_Q_S2 (BINARY_SAMPLER_K_S2 * BARRETT_FACTOR_S2)
#define UNIFORM_Q_S3 (BINARY_SAMPLER_K_S3 * BARRETT_FACTOR_S3)


#define BASE_TABLE_SIZE (4 * CDT_ENTRY_SIZE)
#define BERNOULLI_TABLE_SIZE (4 * BERNOULLI_ENTRY_SIZE)

/* CDT table */
static const __m256i V_CDT[][2] = {{{2200310400551559144, 2200310400551559144, 2200310400551559144, 2200310400551559144}, {3327841033070651387, 3327841033070651387, 3327841033070651387, 3327841033070651387}},
{{7912151619254726620, 7912151619254726620, 7912151619254726620, 7912151619254726620}, {380075531178589176, 380075531178589176, 380075531178589176, 380075531178589176}},
{{5167367257772081627, 5167367257772081627, 5167367257772081627, 5167367257772081627}, {11604843442081400, 11604843442081400, 11604843442081400, 11604843442081400}},
{{5081592746475748971, 5081592746475748971, 5081592746475748971, 5081592746475748971}, {90134450315532, 90134450315532, 90134450315532, 90134450315532}},
{{6522074513864805092, 6522074513864805092, 6522074513864805092, 6522074513864805092}, {175786317361, 175786317361, 175786317361, 175786317361}},
{{2579734681240182346, 2579734681240182346, 2579734681240182346, 2579734681240182346}, {85801740, 85801740, 85801740, 85801740}},
{{8175784047440310133, 8175784047440310133, 8175784047440310133, 8175784047440310133}, {10472, 10472, 10472, 10472}},
{{2947787991558061753, 2947787991558061753, 2947787991558061753, 2947787991558061753}, {0, 0, 0, 0}},
{{22489665999543, 22489665999543, 22489665999543, 22489665999543}, {0, 0, 0, 0}}};

static const uint64_t CDT[9][2] = {
	{2200310400551559144, 3327841033070651387},
	{7912151619254726620,380075531178589176},
	{5167367257772081627,11604843442081400},
	{5081592746475748971,90134450315532},
	{6522074513864805092,175786317361},
	{2579734681240182346,85801740},
	{8175784047440310133,10472},
	{2947787991558061753, 0},
	{22489665999543, 0}
};

static const __m256i V_CDT_LOW_MASK = {CDT_LOW_MASK, CDT_LOW_MASK, CDT_LOW_MASK, CDT_LOW_MASK};

static const __m256i V_K_K_K_K_S1 = {BINARY_SAMPLER_K_S1, BINARY_SAMPLER_K_S1, BINARY_SAMPLER_K_S1, BINARY_SAMPLER_K_S1};
static const __m256i V_K_K_K_K_S2 = {BINARY_SAMPLER_K_S2, BINARY_SAMPLER_K_S2, BINARY_SAMPLER_K_S2, BINARY_SAMPLER_K_S2};
static const __m256i V_K_K_K_K_S3 = {BINARY_SAMPLER_K_S3, BINARY_SAMPLER_K_S3, BINARY_SAMPLER_K_S3, BINARY_SAMPLER_K_S3};


/* coefficients of the exp evaluation polynomial */
static const __m256i EXP_COFF[] = {{0x3e833b70ffa2c5d4, 0x3e833b70ffa2c5d4, 0x3e833b70ffa2c5d4, 0x3e833b70ffa2c5d4},
								   {0x3eb4a480fda7e6e1, 0x3eb4a480fda7e6e1, 0x3eb4a480fda7e6e1, 0x3eb4a480fda7e6e1},
								   {0x3ef01b254493363f, 0x3ef01b254493363f, 0x3ef01b254493363f, 0x3ef01b254493363f},
								   {0x3f242e0e0aa273cc, 0x3f242e0e0aa273cc, 0x3f242e0e0aa273cc, 0x3f242e0e0aa273cc},
								   {0x3f55d8a2334ed31b, 0x3f55d8a2334ed31b, 0x3f55d8a2334ed31b, 0x3f55d8a2334ed31b},
								   {0x3f83b2aa56db0f1a, 0x3f83b2aa56db0f1a, 0x3f83b2aa56db0f1a, 0x3f83b2aa56db0f1a},
								   {0x3fac6b08e11fc57e, 0x3fac6b08e11fc57e, 0x3fac6b08e11fc57e, 0x3fac6b08e11fc57e},
								   {0x3fcebfbdff556072, 0x3fcebfbdff556072, 0x3fcebfbdff556072, 0x3fcebfbdff556072},
								   {0x3fe62e42fefa7fe6, 0x3fe62e42fefa7fe6, 0x3fe62e42fefa7fe6, 0x3fe62e42fefa7fe6},
								   {0x3ff0000000000000, 0x3ff0000000000000, 0x3ff0000000000000, 0x3ff0000000000000}};

/* coefficients of the exp evaluation polynomial */
static const uint64_t EXP_COFF64[10] = {0x3e833b70ffa2c5d4,
									0x3eb4a480fda7e6e1,
									0x3ef01b254493363f,
									0x3f242e0e0aa273cc,
									0x3f55d8a2334ed31b,
									0x3f83b2aa56db0f1a,
									0x3fac6b08e11fc57e,
									0x3fcebfbdff556072,
									0x3fe62e42fefa7fe6,
									0x3ff0000000000000};		
								   
static const __m256d V_INT64_DOUBLE = {0x0010000000000000, 0x0010000000000000, 0x0010000000000000, 0x0010000000000000};
static const __m256d V_DOUBLE_INT64 = {0x0018000000000000, 0x0018000000000000, 0x0018000000000000, 0x0018000000000000};

static const __m256i V_EXP_MANTISSA_MASK = {EXP_MANTISSA_MASK, EXP_MANTISSA_MASK, EXP_MANTISSA_MASK, EXP_MANTISSA_MASK};
static const __m256i V_RES_MANTISSA = {1LL << EXP_MANTISSA_PRECISION, 1LL << EXP_MANTISSA_PRECISION, 1LL << EXP_MANTISSA_PRECISION, 1LL << EXP_MANTISSA_PRECISION};
static const uint64_t V_RES_MANTISSA64 = 1LL << EXP_MANTISSA_PRECISION;
static const __m256i V_RES_EXPONENT = {R_EXPONENT_L - 1023 + 1, R_EXPONENT_L - 1023 + 1, R_EXPONENT_L - 1023 + 1, R_EXPONENT_L - 1023 + 1};
static const uint64_t V_RES_EXPONENT64 = R_EXPONENT_L - 1023 + 1;
static const __m256i V_R_MANTISSA_MASK = {R_MANTISSA_MASK, R_MANTISSA_MASK, R_MANTISSA_MASK, R_MANTISSA_MASK};
static const __m256i V_1 = {1, 1, 1, 1};
static const __m256i V_DOUBLE_ONE = {DOUBLE_ONE, DOUBLE_ONE, DOUBLE_ONE, DOUBLE_ONE};

static const __m256d V_K_2_INV_S1 = {BINARY_SAMPLER_K_2_INV_S1, BINARY_SAMPLER_K_2_INV_S1, BINARY_SAMPLER_K_2_INV_S1, BINARY_SAMPLER_K_2_INV_S1};
static const __m256d V_K_2_INV_S2 = {BINARY_SAMPLER_K_2_INV_S2, BINARY_SAMPLER_K_2_INV_S2, BINARY_SAMPLER_K_2_INV_S2, BINARY_SAMPLER_K_2_INV_S2};
static const __m256d V_K_2_INV_S3 = {BINARY_SAMPLER_K_2_INV_S3, BINARY_SAMPLER_K_2_INV_S3, BINARY_SAMPLER_K_2_INV_S3, BINARY_SAMPLER_K_2_INV_S3};

/* constant time CDT sampler */
static inline __m256i cdt_sampler(unsigned char *r)
{
	__m256i x = _mm256_setzero_si256();
	__m256i r1, r2;
	__m256i r1_lt_cdt0, r2_lt_cdt1;
	__m256i r2_eq_cdt1;
	__m256i b;
	
	uint32_t i;
	
	r1 = _mm256_loadu_si256((__m256i *)r);
	r2 = _mm256_loadu_si256((__m256i *)(r + 32));
	
	r1 = _mm256_and_si256(r1, V_CDT_LOW_MASK);
	r2 = _mm256_and_si256(r2, V_CDT_LOW_MASK);
	// print256_num(r1);	print256_num(r2);
	for (i = 0; i < CDT_LENGTH; i++)
	{
		r1_lt_cdt0 = _mm256_sub_epi64(r1, V_CDT[i][0]);		
		r2_lt_cdt1 = _mm256_sub_epi64(r2, V_CDT[i][1]);
		r2_eq_cdt1 = _mm256_cmpeq_epi64(r2, V_CDT[i][1]);		

		b = _mm256_and_si256(r1_lt_cdt0, r2_eq_cdt1);
		b = _mm256_or_si256(b, r2_lt_cdt1);
		b = _mm256_srli_epi64(b, 63);

		x = _mm256_add_epi64(x, b);
		// printf("\n%u:\t ", i);
		// print256_num(b); print256_num(x);
	}
	
	return x;
}

/* constant time CDT sampler */
static inline void cdt_sampler64(unsigned char *r, uint64_t *x)
{
	// __m256i x = _mm256_setzero_si256();
	// uint64_t x[4] = {0};
	uint64_t r1[4] = {0};	uint64_t r2[4] = {0};
	uint64_t r1_lt_cdt0[4] = {0};	uint64_t r2_lt_cdt1[4] = {0};
	uint64_t r2_eq_cdt1[4] = {0};
	uint64_t b[4] = {0};
	int64_t *p;
	int i, j;
	
	for(i=0; i<4; i++){ 
		p = (int64_t*) &r[i*8];
		r1[i] = *p;
	}
	for(i=0; i<4; i++){ 
		p = (int64_t*) &r[i*8+32];
		r2[i] = *p;
	}
	for(i=0; i<4; i++)	r1[i] &= CDT_LOW_MASK;	
	for(i=0; i<4; i++)	r2[i] &= CDT_LOW_MASK;
	// printf("-%.16lx %.16lx %.16lx %.16lx\n", r1[3], r1[2], r1[1], r1[0]); 
	// printf("-%.16lx %.16lx %.16lx %.16lx\n", r2[3], r2[2], r2[1], r2[0]); 

	for (i = 0; i < CDT_LENGTH; i++)
	{		
		for(j=0; j<4; j++) r1_lt_cdt0[j] = r1[j] - CDT[i][0];
		for(j=0; j<4; j++) r2_lt_cdt1[j] = r2[j] - CDT[i][1];
		for(j=0; j<4; j++) {
			if(r2[i] == CDT[i][1]) r2_eq_cdt1[i] = 0xFFFFFFFFFFFFFFFF;
			else r2_eq_cdt1[i] = 0;
		}
		for(j=0; j<4; j++) r1_lt_cdt0[j] &= r2_eq_cdt1[j];
		for(j=0; j<4; j++) b[j] |= r2_lt_cdt1[j];			
		for(j=0; j<4; j++) b[j] >>= 63;	
		for(j=0; j<4; j++) x[j] += b[j];	
		// printf("\n%u:\t ", i);			
	// printf("%.16lx %.16lx %.16lx %.16lx\n", b[3], b[2], b[1], b[0]); 
	// printf("%.16lx %.16lx %.16lx %.16lx\n", x[3], x[2], x[1], x[0]); 
	}		
}

/* constant time Bernoulli sampler
 * we directly compute exp(-x/(2*sigma_0^2)), 
 * since sigma_0=sqrt(1/2ln2), exp(-x/(2*sigma_0^2))=2^(-x/k^2), 
 * we use a polynomial to directly evaluate 2^(-x/k^2) */
// static const __m256i test[] = {{0x3FF0000000000001, 0x001000000000019f, 0x001000000000019f, 0x433000000000019f}};
static inline void bernoulli_sampler_S1(uint64_t *b, __m256i x, unsigned char *r)
{	
	__m256d vx, vx_1, vx_2, vsum;
	__m256i vt, k, vres, vres_mantissa, vres_exponent, vr_mantissa, vr_exponent, vr_exponent2, vres_eq_1, vr_lt_vres_mantissa, vr_lt_vres_exponent;

	/* 2^x=2^(floor(x)+a)=2^(floor(x))*2^a, where a is in [0,1]
	 * we only evaluate 2^a by using a polynomial */
	x = _mm256_or_si256(x, _mm256_castpd_si256(V_INT64_DOUBLE));
	vx = _mm256_sub_pd(_mm256_castsi256_pd(x), V_INT64_DOUBLE);
	vx = _mm256_mul_pd(vx, V_K_2_INV_S1);
	// print256_num(x);
	vx_1 = _mm256_floor_pd(vx);
	vx_2 = _mm256_add_pd(vx_1, V_DOUBLE_INT64);
	vt = _mm256_sub_epi64(_mm256_castpd_si256(vx_2), _mm256_castpd_si256(V_DOUBLE_INT64));	
	vt = _mm256_slli_epi64(vt, 52);
	// print256_num(vt);
	// print256_num_d(vx_2);
	// /* evaluate 2^a */
	// /* evaluate 2^a */
	vx_2 = _mm256_sub_pd(vx, vx_1);
	vsum = _mm256_fmadd_pd(_mm256_castsi256_pd(EXP_COFF[0]), vx_2, _mm256_castsi256_pd(EXP_COFF[1]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[2]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[3]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[4]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[5]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[6]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[7]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[8]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[9]));

	// print256_num_d(vsum);

	/* combine to compute 2^x */
	vres = _mm256_add_epi64(vt, _mm256_castpd_si256(vsum));
	
	// /* compute the Bernoulli value */
	vres_mantissa = _mm256_and_si256(vres, V_EXP_MANTISSA_MASK);
	vres_mantissa = _mm256_or_si256(vres_mantissa, V_RES_MANTISSA);

	vres_exponent = _mm256_srli_epi64(vres, EXP_MANTISSA_PRECISION);
	vres_exponent = _mm256_add_epi64(vres_exponent, V_RES_EXPONENT);
	vres_exponent = _mm256_sllv_epi64(V_1, vres_exponent);
	// print256_num_i(vres_exponent);
	vr_mantissa = _mm256_loadu_si256((__m256i *)r);
	vr_exponent = _mm256_srli_epi64(vr_mantissa, R_MANTISSA_PRECISION);
	vr_mantissa = _mm256_and_si256(vr_mantissa, V_R_MANTISSA_MASK);
	vr_exponent2 = _mm256_set_epi64x(r[35], r[34], r[33], r[32]);
	vr_exponent2 = _mm256_slli_epi64(vr_exponent2, 64 - R_MANTISSA_PRECISION);
	vr_exponent = _mm256_or_si256(vr_exponent, vr_exponent2);
	// print256_num(vr_exponent);
	/* (res == 1.0) || ((r_mantissa < res_mantissa) && (r_exponent < (1 << res_exponent))) */
	vres_eq_1 = _mm256_cmpeq_epi64(vres, V_DOUBLE_ONE);
	vr_lt_vres_mantissa = _mm256_sub_epi64(vr_mantissa, vres_mantissa);	
	vr_lt_vres_exponent = _mm256_sub_epi64(vr_exponent, vres_exponent);
	// print256_num(vr_lt_vres_mantissa);
	k = _mm256_and_si256(vr_lt_vres_mantissa, vr_lt_vres_exponent);
	k = _mm256_or_si256(k, vres_eq_1);

	_mm256_store_si256((__m256i *)(b), k);
	// print256_num(k);
}

static inline void bernoulli_sampler_S1_64(uint64_t *b, uint64_t *x, unsigned char *r)
{	
	int i=0;
	double vx64[4] = {0}, vx1_64[4] = {0}, vx2_64[4] = {0}, vsum64[4] = {0};
	int64_t vt64[4] = {0}, vres64[4] = {0}, vres_mantissa64[4] = {0}, vres_exponent64[4] = {0}, vr_mantissa64[4] = {0}, vr_exponent64[4] = {0}, vr_exponent2_64[4] = {0}, vres_eq_164[4] = {0}, vr_lt_vres_mantissa64[4] = {0}, vr_lt_vres_exponent64[4] = {0};
	int64_t mask = 0x4330000000000000, *p;
	__m256d vx, vx_1, vx_2, vsum;
	__m256i vt, k, vres, vres_mantissa, vres_exponent, vr_mantissa, vr_exponent, vr_exponent2, vres_eq_1, vr_lt_vres_mantissa, vr_lt_vres_exponent;

	/* 2^x=2^(floor(x)+a)=2^(floor(x))*2^a, where a is in [0,1]
	 * we only evaluate 2^a by using a polynomial */
	// x = _mm256_or_si256(x, _mm256_castpd_si256(V_INT64_DOUBLE));
	// vx = _mm256_sub_pd(_mm256_castsi256_pd(x), V_INT64_DOUBLE);
	// vx = _mm256_mul_pd(vx, V_K_2_INV_S1);
	p = (uint64_t*) &mask;		

	for(i=0; i<4; i++) x[i] |= *p;
	for(i=0; i<4; i++) vx64[i] = (double) (x[i] - 0x4330000000000000);
	for(i=0; i<4; i++) vx64[i] = vx64[i] * BINARY_SAMPLER_K_2_INV_S1;
	// printf("%.16lx %.16lx %.16lx %.16lx\n", x[3], x[2], x[1], x[0]);	
	
	for(i=0; i<4; i++) vx1_64[i] = floor(vx64[i]);		
	for(i=0; i<4; i++) vx2_64[i] = vx1_64[i] + 0x0018000000000000;
	for(i=0; i<4; i++) vt64[i] = (int64_t) vx2_64[i] - 0x0018000000000000;
	for(i=0; i<4; i++) vt64[i] = vt64[i] <<52;
	// printf("%.2f %.2f %.2f %.2f\n", vx2_64[3], vx2_64[2], vx2_64[1], vx2_64[0]);
	// printf("%.16ld %.16ld %.16ld %.16ld\n", vt64[3], vt64[2], vt64[1], vt64[0]);
	// printf("%.16lx %.16lx %.16lx %.16lx\n", vt64[3], vt64[2], vt64[1], vt64[0]);	
	// /* evaluate 2^a */
	double *g1, *g2 ;

	for(i=0; i<4; i++) vx2_64[i] = vx64[i] - vx1_64[i];
	g1 = (double *) &EXP_COFF64[0];	g2 = (double *) &EXP_COFF64[1];	
	for(i=0; i<4; i++) vsum64[i] = (*g1 * vx2_64[i]) + *g2;		
	g2 = (double *) &EXP_COFF64[2];	
	for(i=0; i<4; i++) vsum64[i] = (vsum64[i] * vx2_64[i]) + *g2;	
	g2 = (double *) &EXP_COFF64[3];	
	for(i=0; i<4; i++) vsum64[i] = (vsum64[i] * vx2_64[i]) + *g2;	
	g2 = (double *) &EXP_COFF64[4];	
	for(i=0; i<4; i++) vsum64[i] = (vsum64[i] * vx2_64[i]) + *g2;	
	g2 = (double *) &EXP_COFF64[5];	
	for(i=0; i<4; i++) vsum64[i] = (vsum64[i] * vx2_64[i]) + *g2;	
	g2 = (double *) &EXP_COFF64[6];	
	for(i=0; i<4; i++) vsum64[i] = (vsum64[i] * vx2_64[i]) + *g2;	
	g2 = (double *) &EXP_COFF64[7];	
	for(i=0; i<4; i++) vsum64[i] = (vsum64[i] * vx2_64[i]) + *g2;	
	g2 = (double *) &EXP_COFF64[8];	
	for(i=0; i<4; i++) vsum64[i] = (vsum64[i] * vx2_64[i]) + *g2;	
	g2 = (double *) &EXP_COFF64[9];	
	for(i=0; i<4; i++) vsum64[i] = (vsum64[i] * vx2_64[i]) + *g2;		

	// printf("%.16f %.16f %.16f %.16f\n", vsum64[3], vsum64[2], vsum64[1], vsum64[0]);
	// /* combine to compute 2^x */
	// vres = _mm256_add_epi64(vt, _mm256_castpd_si256(vsum));
	for(i=0; i<4; i++) {
		p = (uint64_t*) &vsum64[i];
		vres64[i] = vt64[i] + *p;
	}

	// /* compute the Bernoulli value */

	for(i=0; i<4; i++) vres_mantissa64[i] = vres64[i] & EXP_MANTISSA_MASK;
	for(i=0; i<4; i++) vres_mantissa64[i] |= V_RES_MANTISSA64;
	for(i=0; i<4; i++) vres_exponent64[i] = vres64[i] >> EXP_MANTISSA_PRECISION;
	for(i=0; i<4; i++) vres_exponent64[i] += V_RES_EXPONENT64;
	for(i=0; i<4; i++) vres_exponent64[i] = 1LL << vres_exponent64[i];		
	// printf("%.16ld %.16ld %.16ld %.16ld\n", vres_exponent64[3], vres_exponent64[2], vres_exponent64[1], vres_exponent64[0]);
	for(i=0; i<4; i++){ 
		p = (int64_t*) &r[i*8];
		vr_mantissa64[i] = *p;
	}
	for(i=0; i<4; i++) vr_exponent64[i] = (vr_mantissa64[i] >> R_MANTISSA_PRECISION) & 0x00000000000007ff;
	for(i=0; i<4; i++) vr_mantissa64[i] &= R_MANTISSA_MASK;
	// printf("%.16lx %.16lx %.16lx %.16lx\n", vr_mantissa64[3], vr_mantissa64[2], vr_mantissa64[1], vr_mantissa64[0]);
	
	for(i=0; i<4; i++) vr_exponent2_64[i] = r[32+i];
	for(i=0; i<4; i++) vr_exponent2_64[i] <<= (64 - R_MANTISSA_PRECISION);
	for(i=0; i<4; i++) vr_exponent64[i]	|= vr_exponent2_64[i];	
	// printf("%.16lx %.16lx %.16lx %.16lx\n", vr_exponent64[3], vr_exponent64[2], vr_exponent64[1], vr_exponent64[0]);		
	/* (res == 1.0) || ((r_mantissa < res_mantissa) && (r_exponent < (1 << res_exponent))) */
	for(i=0; i<4; i++) {
		if(vres64[i] == DOUBLE_ONE) vres_eq_164[i] = 0xffffffffffffffff;
		else vres_eq_164[i] = 0;
	}
	for(i=0; i<4; i++) vr_lt_vres_mantissa64[i] = vr_mantissa64[i] - vres_mantissa64[i];
	for(i=0; i<4; i++) vr_lt_vres_exponent64[i] = vr_exponent64[i] - vres_exponent64[i];		
	// printf("%.16lx %.16lx %.16lx %.16lx\n", vr_lt_vres_mantissa64[3], vr_lt_vres_mantissa64[2], vr_lt_vres_mantissa64[1], vr_lt_vres_mantissa64[0]);			
	for(i=0; i<4; i++) b[i] =  vr_lt_vres_mantissa64[i] & vr_lt_vres_exponent64[i];
	for(i=0; i<4; i++) b[i] |= vres_eq_164[i];		
	// _mm256_store_si256((__m256i *)(b), k);
	// printf("%.16lx %.16lx %.16lx %.16lx\n", vres_eq_164[3], vres_eq_164[2], vres_eq_164[1], vres_eq_164[0]);	
	// printf("%.16lx %.16lx %.16lx %.16lx\n", b[3], b[2], b[1], b[0]);				
}


static inline void bernoulli_sampler_S2(uint64_t *b, __m256i x, unsigned char *r)
{	
	__m256d vx, vx_1, vx_2, vsum;
	__m256i vt, k, vres, vres_mantissa, vres_exponent, vr_mantissa, vr_exponent, vr_exponent2, vres_eq_1, vr_lt_vres_mantissa, vr_lt_vres_exponent;

	/* 2^x=2^(floor(x)+a)=2^(floor(x))*2^a, where a is in [0,1]
	 * we only evaluate 2^a by using a polynomial */
	x = _mm256_or_si256(x, _mm256_castpd_si256(V_INT64_DOUBLE));
	vx = _mm256_sub_pd(_mm256_castsi256_pd(x), V_INT64_DOUBLE);
	vx = _mm256_mul_pd(vx, V_K_2_INV_S2);
	
	vx_1 = _mm256_floor_pd(vx);
	vx_2 = _mm256_add_pd(vx_1, V_DOUBLE_INT64);
	vt = _mm256_sub_epi64(_mm256_castpd_si256(vx_2), _mm256_castpd_si256(V_DOUBLE_INT64));	
	vt = _mm256_slli_epi64(vt, 52);
	
	/* evaluate 2^a */
	vx_2 = _mm256_sub_pd(vx, vx_1);
	vsum = _mm256_fmadd_pd(_mm256_castsi256_pd(EXP_COFF[0]), vx_2, _mm256_castsi256_pd(EXP_COFF[1]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[2]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[3]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[4]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[5]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[6]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[7]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[8]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[9]));
	
	/* combine to compute 2^x */
	vres = _mm256_add_epi64(vt, _mm256_castpd_si256(vsum));

	/* compute the Bernoulli value */
	vres_mantissa = _mm256_and_si256(vres, V_EXP_MANTISSA_MASK);
	vres_mantissa = _mm256_or_si256(vres_mantissa, V_RES_MANTISSA);
	
	vres_exponent = _mm256_srli_epi64(vres, EXP_MANTISSA_PRECISION);
	vres_exponent = _mm256_add_epi64(vres_exponent, V_RES_EXPONENT);
	vres_exponent = _mm256_sllv_epi64(V_1, vres_exponent);
	
	vr_mantissa = _mm256_loadu_si256((__m256i *)r);
	vr_exponent = _mm256_srli_epi64(vr_mantissa, R_MANTISSA_PRECISION);
	vr_mantissa = _mm256_and_si256(vr_mantissa, V_R_MANTISSA_MASK);
	vr_exponent2 = _mm256_set_epi64x(r[35], r[34], r[33], r[32]);
	vr_exponent2 = _mm256_slli_epi64(vr_exponent2, 64 - R_MANTISSA_PRECISION);
	vr_exponent = _mm256_or_si256(vr_exponent, vr_exponent2);

	/* (res == 1.0) || ((r_mantissa < res_mantissa) && (r_exponent < (1 << res_exponent))) */
	vres_eq_1 = _mm256_cmpeq_epi64(vres, V_DOUBLE_ONE);
	vr_lt_vres_mantissa = _mm256_sub_epi64(vr_mantissa, vres_mantissa);	
	vr_lt_vres_exponent = _mm256_sub_epi64(vr_exponent, vres_exponent);
	
	k = _mm256_and_si256(vr_lt_vres_mantissa, vr_lt_vres_exponent);
	k = _mm256_or_si256(k, vres_eq_1);

	_mm256_store_si256((__m256i *)(b), k);
}

static inline void bernoulli_sampler_S2_64(uint64_t *b, uint64_t *x, unsigned char *r)
{	
	int i=0;
	double vx64[4] = {0}, vx1_64[4] = {0}, vx2_64[4] = {0}, vsum64[4] = {0};
	int64_t vt64[4] = {0}, vres64[4] = {0}, vres_mantissa64[4] = {0}, vres_exponent64[4] = {0}, vr_mantissa64[4] = {0}, vr_exponent64[4] = {0}, vr_exponent2_64[4] = {0}, vres_eq_164[4] = {0}, vr_lt_vres_mantissa64[4] = {0}, vr_lt_vres_exponent64[4] = {0};
	// wklee V_INT64_DOUBLE in Hex form
	int64_t mask = 0x4330000000000000, *p;
	double *p2;
	/* 2^x=2^(floor(x)+a)=2^(floor(x))*2^a, where a is in [0,1]
	 * we only evaluate 2^a by using a polynomial */

	p = (uint64_t*) &mask;		

	for(i=0; i<4; i++) x[i] =  x[i]| *p;
	// for(i=0; i<4; i++) vx64[i] = (double) (x[i] - 0x4330000000000000);
	// printf("%.16lx %.16lx %.16lx %.16lx\n", x[3], x[2], x[1], x[0]);			
	
	for(i=0; i<4; i++) 
	{
		p2 = (double*) &x[i];
		vx64[i] = *p2 - 4503599627370496.0;
	}

	// for(i=0; i<4; i++){ 
	// 	x[i] = (x[i] - 0x4330000000000000);
	// 	p2 = (double*) &x[i];
	// 	vx64[i] = *p2;
	// }
	// wklee, sometimes the inputs from x are larger than 53-bit, wrong answer in double precision. Only happens in S2 and S3
	for(i=0; i<4; i++) vx64[i] = vx64[i] * BINARY_SAMPLER_K_2_INV_S2;
	
	for(i=0; i<4; i++) vx1_64[i] = floor(vx64[i]);		
	for(i=0; i<4; i++) vx2_64[i] = vx1_64[i] + 0x0018000000000000;
	for(i=0; i<4; i++) vt64[i] = (int64_t) vx2_64[i] - 0x0018000000000000;
	for(i=0; i<4; i++) vt64[i] = vt64[i] <<52;
	// printf("%.16lx %.16lx %.16lx %.16lx\n", vt64[3], vt64[2], vt64[1], vt64[0]);	
	// /* evaluate 2^a */
	double *g1, *g2 ;

	for(i=0; i<4; i++) vx2_64[i] = vx64[i] - vx1_64[i];
	g1 = (double *) &EXP_COFF64[0];	g2 = (double *) &EXP_COFF64[1];	
	for(i=0; i<4; i++) vsum64[i] = (*g1 * vx2_64[i]) + *g2;		
	g2 = (double *) &EXP_COFF64[2];	
	for(i=0; i<4; i++) vsum64[i] = (vsum64[i] * vx2_64[i]) + *g2;	
	g2 = (double *) &EXP_COFF64[3];	
	for(i=0; i<4; i++) vsum64[i] = (vsum64[i] * vx2_64[i]) + *g2;	
	g2 = (double *) &EXP_COFF64[4];	
	for(i=0; i<4; i++) vsum64[i] = (vsum64[i] * vx2_64[i]) + *g2;	
	g2 = (double *) &EXP_COFF64[5];	
	for(i=0; i<4; i++) vsum64[i] = (vsum64[i] * vx2_64[i]) + *g2;	
	g2 = (double *) &EXP_COFF64[6];	
	for(i=0; i<4; i++) vsum64[i] = (vsum64[i] * vx2_64[i]) + *g2;	
	g2 = (double *) &EXP_COFF64[7];	
	for(i=0; i<4; i++) vsum64[i] = (vsum64[i] * vx2_64[i]) + *g2;	
	g2 = (double *) &EXP_COFF64[8];	
	for(i=0; i<4; i++) vsum64[i] = (vsum64[i] * vx2_64[i]) + *g2;	
	g2 = (double *) &EXP_COFF64[9];	
	for(i=0; i<4; i++) vsum64[i] = (vsum64[i] * vx2_64[i]) + *g2;

	// /* combine to compute 2^x */
	for(i=0; i<4; i++) {
		p = (uint64_t*) &vsum64[i];
		vres64[i] = vt64[i] + *p;
	}

	// /* compute the Bernoulli value */

	for(i=0; i<4; i++) vres_mantissa64[i] = vres64[i] & EXP_MANTISSA_MASK;
	for(i=0; i<4; i++) vres_mantissa64[i] |= V_RES_MANTISSA64;
	for(i=0; i<4; i++) vres_exponent64[i] = vres64[i] >> EXP_MANTISSA_PRECISION;
	for(i=0; i<4; i++) vres_exponent64[i] += V_RES_EXPONENT64;	

	for(i=0; i<4; i++) {
		if(vres_exponent64[i]>=0)
			vres_exponent64[i] = (uint64_t) 1LL << vres_exponent64[i];
		else
			vres_exponent64[i] = 0;
	}
	
	for(i=0; i<4; i++){ 
		p = (int64_t*) &r[i*8];
		vr_mantissa64[i] = *p;
	}
	for(i=0; i<4; i++) vr_exponent64[i] = (vr_mantissa64[i] >> R_MANTISSA_PRECISION) & 0x00000000000007ff;
	for(i=0; i<4; i++) vr_mantissa64[i] &= R_MANTISSA_MASK;

	for(i=0; i<4; i++) vr_exponent2_64[i] = r[32+i];
	for(i=0; i<4; i++) vr_exponent2_64[i] <<= (64 - R_MANTISSA_PRECISION);
	for(i=0; i<4; i++) vr_exponent64[i]	|= vr_exponent2_64[i];	

	/* (res == 1.0) || ((r_mantissa < res_mantissa) && (r_exponent < (1 << res_exponent))) */
	for(i=0; i<4; i++) {
		if(vres64[i] == DOUBLE_ONE) vres_eq_164[i] = 0xffffffffffffffff;
		else vres_eq_164[i] = 0;
	}
	for(i=0; i<4; i++) vr_lt_vres_mantissa64[i] = vr_mantissa64[i] - vres_mantissa64[i];
	for(i=0; i<4; i++) vr_lt_vres_exponent64[i] = vr_exponent64[i] - vres_exponent64[i];		
	
	for(i=0; i<4; i++) b[i] =  vr_lt_vres_mantissa64[i] & vr_lt_vres_exponent64[i];
	for(i=0; i<4; i++) b[i] |= vres_eq_164[i];		
	
	// printf("-%.16lx %.16lx %.16lx %.16lx\n", b[3], b[2], b[1], b[0]);				
}
static inline void bernoulli_sampler_S3(uint64_t *b, __m256i x, unsigned char *r)
{	
	__m256d vx, vx_1, vx_2, vsum;
	__m256i vt, k, vres, vres_mantissa, vres_exponent, vr_mantissa, vr_exponent, vr_exponent2, vres_eq_1, vr_lt_vres_mantissa, vr_lt_vres_exponent;

	/* 2^x=2^(floor(x)+a)=2^(floor(x))*2^a, where a is in [0,1]
	 * we only evaluate 2^a by using a polynomial */
	x = _mm256_or_si256(x, _mm256_castpd_si256(V_INT64_DOUBLE));
	vx = _mm256_sub_pd(_mm256_castsi256_pd(x), V_INT64_DOUBLE);
	vx = _mm256_mul_pd(vx, V_K_2_INV_S3);
	
	vx_1 = _mm256_floor_pd(vx);
	vx_2 = _mm256_add_pd(vx_1, V_DOUBLE_INT64);
	vt = _mm256_sub_epi64(_mm256_castpd_si256(vx_2), _mm256_castpd_si256(V_DOUBLE_INT64));	
	vt = _mm256_slli_epi64(vt, 52);
	
	/* evaluate 2^a */
	vx_2 = _mm256_sub_pd(vx, vx_1);
	vsum = _mm256_fmadd_pd(_mm256_castsi256_pd(EXP_COFF[0]), vx_2, _mm256_castsi256_pd(EXP_COFF[1]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[2]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[3]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[4]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[5]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[6]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[7]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[8]));
	vsum = _mm256_fmadd_pd(vsum, vx_2, _mm256_castsi256_pd(EXP_COFF[9]));
	
	/* combine to compute 2^x */
	vres = _mm256_add_epi64(vt, _mm256_castpd_si256(vsum));

	/* compute the Bernoulli value */
	vres_mantissa = _mm256_and_si256(vres, V_EXP_MANTISSA_MASK);
	vres_mantissa = _mm256_or_si256(vres_mantissa, V_RES_MANTISSA);
	
	vres_exponent = _mm256_srli_epi64(vres, EXP_MANTISSA_PRECISION);
	vres_exponent = _mm256_add_epi64(vres_exponent, V_RES_EXPONENT);
	vres_exponent = _mm256_sllv_epi64(V_1, vres_exponent);
	
	vr_mantissa = _mm256_loadu_si256((__m256i *)r);
	vr_exponent = _mm256_srli_epi64(vr_mantissa, R_MANTISSA_PRECISION);
	vr_mantissa = _mm256_and_si256(vr_mantissa, V_R_MANTISSA_MASK);
	vr_exponent2 = _mm256_set_epi64x(r[35], r[34], r[33], r[32]);
	vr_exponent2 = _mm256_slli_epi64(vr_exponent2, 64 - R_MANTISSA_PRECISION);
	vr_exponent = _mm256_or_si256(vr_exponent, vr_exponent2);

	/* (res == 1.0) || ((r_mantissa < res_mantissa) && (r_exponent < (1 << res_exponent))) */
	vres_eq_1 = _mm256_cmpeq_epi64(vres, V_DOUBLE_ONE);
	vr_lt_vres_mantissa = _mm256_sub_epi64(vr_mantissa, vres_mantissa);	
	vr_lt_vres_exponent = _mm256_sub_epi64(vr_exponent, vres_exponent);
	
	k = _mm256_and_si256(vr_lt_vres_mantissa, vr_lt_vres_exponent);
	k = _mm256_or_si256(k, vres_eq_1);

	_mm256_store_si256((__m256i *)(b), k);
}

/* make sure that Pr(rerun the PRG)<=2^(-64) */
static inline void uniform_sampler_S1_noAVX(unsigned char *r, uint64_t* sam64y1, uint64_t* sam64y2)
{
	uint64_t sample[8] __attribute__ ((aligned (32)));
	uint32_t i = 0, j = 0;
	uint64_t x;
	
	while (j < 8)
	{
		do
		{	/* we ignore the low probability of rerunning the PRG */
			x = *((uint32_t *)(r + UNIFORM_SIZE * (i++)));
		} while (1 ^ ((x - UNIFORM_Q_S1) >> 63));

		x = x - ((((x * BARRETT_FACTOR_S1) >> BARRETT_BITSHIFT) + 1) * BINARY_SAMPLER_K_S1);
		x = x + (x >> 63) * BINARY_SAMPLER_K_S1;
		
		sample[j++] = x;
		//printf("%lu\t", x);
	}

	for(i=0; i<4; i++ ) sam64y1[i] = sample[i];
	for(i=0; i<4; i++ ) sam64y2[i] = sample[i + 4];
	
	// *y1 = _mm256_load_si256((__m256i *)(sample));
	// *y2 = _mm256_load_si256((__m256i *)(sample + 4));
}



/* make sure that Pr(rerun the PRG)<=2^(-64) */
static inline void uniform_sampler_S1(unsigned char *r, __m256i *y1, __m256i *y2)
{
	uint64_t sample[8] __attribute__ ((aligned (32)));
	uint32_t i = 0, j = 0;
	uint64_t x;
	
	while (j < 8)
	{
		do
		{	/* we ignore the low probability of rerunning the PRG */
			x = *((uint32_t *)(r + UNIFORM_SIZE * (i++)));
		} while (1 ^ ((x - UNIFORM_Q_S1) >> 63));

		x = x - ((((x * BARRETT_FACTOR_S1) >> BARRETT_BITSHIFT) + 1) * BINARY_SAMPLER_K_S1);
		x = x + (x >> 63) * BINARY_SAMPLER_K_S1;
		
		sample[j++] = x;
		//printf("%lu\t", x);
	}
	
	*y1 = _mm256_load_si256((__m256i *)(sample));
	*y2 = _mm256_load_si256((__m256i *)(sample + 4));
}

static inline void uniform_sampler_S2_noAVX(unsigned char *r, uint64_t* sam64y1, uint64_t* sam64y2)
{
	uint64_t sample[8] __attribute__ ((aligned (32)));
	uint32_t i = 0, j = 0;
	uint64_t x;
	
	while (j < 8)
	{
		do
		{	/* we ignore the low probability of rerunning the PRG */
			x = *((uint32_t *)(r + UNIFORM_SIZE * (i++)));
		} while (1 ^ ((x - UNIFORM_Q_S2) >> 63));

		x = x - ((((x * BARRETT_FACTOR_S2) >> BARRETT_BITSHIFT) + 1) * BINARY_SAMPLER_K_S2);
		x = x + (x >> 63) * BINARY_SAMPLER_K_S2;
		
		sample[j++] = x;
		//printf("%lu\t", x);
	}

	for(i=0; i<4; i++ ) sam64y1[i] = sample[i];
	for(i=0; i<4; i++ ) sam64y2[i] = sample[i + 4];
}

static inline void uniform_sampler_S2(unsigned char *r, __m256i *y1, __m256i *y2)
{
	uint64_t sample[8] __attribute__ ((aligned (32)));
	uint32_t i = 0, j = 0;
	uint64_t x;
	
	while (j < 8)
	{
		do
		{	/* we ignore the low probability of rerunning the PRG */
			x = *((uint32_t *)(r + UNIFORM_SIZE * (i++)));
		} while (1 ^ ((x - UNIFORM_Q_S2) >> 63));

		x = x - ((((x * BARRETT_FACTOR_S2) >> BARRETT_BITSHIFT) + 1) * BINARY_SAMPLER_K_S2);
		x = x + (x >> 63) * BINARY_SAMPLER_K_S2;
		
		sample[j++] = x;
		//printf("%lu\t", x);
	}
	
	*y1 = _mm256_load_si256((__m256i *)(sample));
	*y2 = _mm256_load_si256((__m256i *)(sample + 4));
}

static inline void uniform_sampler_S3(unsigned char *r, __m256i *y1, __m256i *y2)
{
	uint64_t sample[8] __attribute__ ((aligned (32)));
	uint32_t i = 0, j = 0;
	uint64_t x;
	
	while (j < 8)
	{
		do
		{	/* we ignore the low probability of rerunning the PRG */
			x = *((uint32_t *)(r + UNIFORM_SIZE * (i++)));
		} while (1 ^ ((x - UNIFORM_Q_S3) >> 63));

		x = x - ((((x * BARRETT_FACTOR_S3) >> BARRETT_BITSHIFT) + 1) * BINARY_SAMPLER_K_S3);
		x = x + (x >> 63) * BINARY_SAMPLER_K_S3;
		
		sample[j++] = x;
		//printf("%lu\t", x);
	}
	
	*y1 = _mm256_load_si256((__m256i *)(sample));
	*y2 = _mm256_load_si256((__m256i *)(sample + 4));
}

void gaussian_sampler_S1_ori(aes256ctr_ctx *state, uint32_t sample[SIFE_NMODULI][SIFE_N], uint32_t slen)
{
	__m256i v_x, v_y1, v_y2, v_z, v_b_in;
	uint64_t z[8] __attribute__ ((aligned (32)));
	uint64_t b[8] __attribute__ ((aligned (32)));
	
	//unsigned char r[2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE) + UNIFORM_REJ * UNIFORM_SIZE + 1] __attribute__ ((aligned (32)));
	unsigned char r[384];
	unsigned char *r1;
	
	uint32_t i = 8, j = 0;
	uint64_t k;
	//aes256ctr_ctx state;
	//aes256ctr_init(&state, seed, 0);
	
	const uint32_t AES_ROUNDS=3;

	uint64_t mod;
	uint32_t mod1, mod2, mod3;

	#if SEC_LEVEL==2
	uint32_t mod4;
	#endif

	while (j < slen)
	{
		do
		{
			if (i == 8)
			{
				/* x<--D_sigma_0, y<--U([0,k-1]), z=kx+y */
				//fastrandombytes(r, 2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE) + UNIFORM_REJ * UNIFORM_SIZE + 1);
				//aes256ctr_squeezeblocks(r, AES_ROUNDS, &state);
				aes256ctr_squeezeblocks(r, AES_ROUNDS, state);
				
				uniform_sampler_S1(r + 2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE), &v_y1, &v_y2);
				// print256_num(v_y1);	print256_num(v_y2);
				r1 = r;
				v_x = cdt_sampler(r1);
				v_x = _mm256_mul_epu32(v_x, V_K_K_K_K_S1);
				v_z = _mm256_add_epi64(v_x, v_y1);
				_mm256_store_si256((__m256i *)(z), v_z);
				/* b<--Bernoulli(exp(-y(y+2kx)/2sigma_0^2)) */
				v_b_in = _mm256_add_epi64(v_z, v_x);
				v_b_in = _mm256_mul_epu32(v_b_in, v_y1);
				bernoulli_sampler_S1(b, v_b_in, r1 + BASE_TABLE_SIZE);
				
				r1 = r + (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE);
				v_x = cdt_sampler(r1);
				v_x = _mm256_mul_epu32(v_x, V_K_K_K_K_S1);
				v_z = _mm256_add_epi64(v_x, v_y2);
				_mm256_store_si256((__m256i *)(z + 4), v_z);
				/* b<--Bernoulli(exp(-y(y+2kx)/2sigma_0^2)) */
				v_b_in = _mm256_add_epi64(v_z, v_x);
				v_b_in = _mm256_mul_epu32(v_b_in, v_y2);
				bernoulli_sampler_S1(b + 4, v_b_in, r1 + BASE_TABLE_SIZE);

				i = 0;
			}
			
			k = (r[2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE) + UNIFORM_REJ * UNIFORM_SIZE] >> i) & 0x1;
			i++;			
		} while (1 ^ ((b[i - 1] & ((z[i - 1] | -z[i - 1]) | (k | -k))) >> 63)); /* rejection condition: b=0 or ((b=1) && (z=0) && (k=0)) */

		mod=z[i-1];

		/*
		if(mod>10*225){
			printf("\n----LARGE----\n");
		}
		*/

		mod1=mod_prime(mod, 0);
		mod2=mod_prime(mod, 1);
		mod3=mod_prime(mod, 2);		
		
		sample[0][j]=(1-k)*mod1+k*mod_prime(SIFE_MOD_Q_I[0]-mod1, 0);
		sample[1][j]=(1-k)*mod2+k*mod_prime(SIFE_MOD_Q_I[1]-mod2, 1);
		sample[2][j]=(1-k)*mod3+k*mod_prime(SIFE_MOD_Q_I[2]-mod3, 2);


		#if SEC_LEVEL==2
			mod4=mod_prime(mod, 3);	
			sample[3][j]=(1-k)*mod4+k*mod_prime(SIFE_MOD_Q_I[3]-mod4, 3);
		#endif

		j++;

	}

}


//void gaussian_sampler_S1(unsigned char *seed, int64_t sample[SIFE_NMODULI][SIFE_N], uint32_t slen)
void gaussian_sampler_S1(aes256ctr_ctx *state, uint32_t sample[SIFE_NMODULI][SIFE_N], uint32_t slen)
{
	__m256i v_x, v_y1, v_y2, v_z, v_b_in;
	uint64_t z[8] __attribute__ ((aligned (32)));
	uint64_t b[8] __attribute__ ((aligned (32)));
	uint64_t b64[8] __attribute__ ((aligned (32)));
	uint64_t v64_y1[8] __attribute__ ((aligned (32)));
	uint64_t v64_y2[8] __attribute__ ((aligned (32)));
	uint64_t vx64[4] ={0};		uint64_t vb_in_64[4] ={0};	

	//unsigned char r[2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE) + UNIFORM_REJ * UNIFORM_SIZE + 1] __attribute__ ((aligned (32)));
	unsigned char r[384] = {0};
	unsigned char *r1;
	
	uint32_t i = 8, j = 0, l = 0;
	uint64_t k;
	//aes256ctr_ctx state;
	//aes256ctr_init(&state, seed, 0);
	
	const uint32_t AES_ROUNDS=3;

	uint64_t mod;
	uint32_t mod1, mod2, mod3;

	#if SEC_LEVEL==2
	uint32_t mod4;
	#endif

	while (j < slen)
	{
		do
		{			
			if (i == 8)
			{
				for(l=0; l<4; l++) vx64[l] = 0;
				/* x<--D_sigma_0, y<--U([0,k-1]), z=kx+y */
				//fastrandombytes(r, 2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE) + UNIFORM_REJ * UNIFORM_SIZE + 1);
				//aes256ctr_squeezeblocks(r, AES_ROUNDS, &state);
				aes256ctr_squeezeblocks(r, AES_ROUNDS, state);
				// for (k = 0; k < 384; k++) printf("%x ", r[k]);
				// uniform_sampler_S1(r + 2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE), &v_y1, &v_y2);
				uniform_sampler_S1_noAVX(r + 2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE), v64_y1, v64_y2);
				
				// for (k = 0; k < 8; k++) printf("%lx ", v64_y1[k]); printf("\n");
				// for (k = 0; k < 8; k++) printf("%lx ", v64_y2[k]);printf("\n");
				// print256_num(v_y1);
    			// printf("%.16lx %.16lx %.16lx %.16lx\n", v64_y1[3], v64_y1[2], v64_y1[1], v64_y1[0]);
    			// print256_num(v_y2);
    			// printf("%.16lx %.16lx %.16lx %.16lx\n", v64_y2[3], v64_y2[2], v64_y2[1], v64_y2[0]);    			
				r1 = r;
				// v_x = cdt_sampler(r1);
				cdt_sampler64(r1, vx64);				    			
				// for (k = 0; k < 4; k++) printf("%lx ", vx64[k]);printf("\n");
				// v_x = _mm256_mul_epu32(v_x, V_K_K_K_K_S1);
				for(l=0; l<4; l++) vx64[l] = (uint32_t) vx64[l] * BINARY_SAMPLER_K_S1;				
				// v_z = _mm256_add_epi64(v_x, v_y1);
				for(l=0; l<4; l++) z[l] = vx64[l] + v64_y1[l];		
				// for (k = 0; k < 4; k++) printf("%lx ", z[k]);printf("\n");
				// _mm256_store_si256((__m256i *)(z), v_z);
				// print256_num(v_x);state
				// printf("%.16lx %.16lx %.16lx %.16lx\n", vx64[3], vx64[2], vx64[1], vx64[0]);	
				// /* b<--Bernoulli(exp(-y(y+2kx)/2sigma_0^2)) */
				// v_b_in = _mm256_add_epi64(v_z, v_x);
				// v_b_in = _mm256_mul_epu32(v_b_in, v_y1);				
				for(l=0; l<4; l++) vb_in_64[l] = z[l] + vx64[l];
				for(l=0; l<4; l++) vb_in_64[l] = (uint32_t) vb_in_64[l] * v64_y1[l];
				// if(j==16) print256_num(v_b_in);
				// printf("%.16lx %.16lx %.16lx %.16lx\n", vb_in_64[3], vb_in_64[2], vb_in_64[1], vb_in_64[0]);	
				// printf("\nj: %u\n", j);
				// bernoulli_sampler_S1(b, v_b_in, r1 + BASE_TABLE_SIZE);
				bernoulli_sampler_S1_64(b, vb_in_64, r1 + BASE_TABLE_SIZE);
	
				// for(l=0; l<4; l++) {printf("** "); printf("%.16lx ", b[l]);} printf("\n");
				// if(j==16) for(l=0; l<4; l++) {printf("** "); printf("%.16lx ", b64[l]);} printf("\n");		
				r1 = r + (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE);
				
				// v_x = cdt_sampler(r1);
				for(l=0; l<4; l++) vx64[l] = 0;
				cdt_sampler64(r1, vx64);	
				// printf("%.16lx %.16lx %.16lx %.16lx\n", vx64[3], vx64[2], vx64[1], vx64[0]);	
				// v_x = _mm256_mul_epu32(v_x, V_K_K_K_K_S1);
				for(l=0; l<4; l++) vx64[l] = (uint32_t) vx64[l] * BINARY_SAMPLER_K_S1;		
				// v_z = _mm256_add_epi64(v_x, v_y2);
				for(l=0; l<4; l++) z[l+4] = vx64[l] + v64_y2[l];
				// print256_num(v_z);
    			// printf("%.16lx %.16lx %.16lx %.16lx\n", z[4+3], z[4+2], z[4+1], z[4+0]);
				// // _mm256_store_si256((__m256i *)(z + 4), v_z);
				// /* b<--Bernoulli(exp(-y(y+2kx)/2sigma_0^2)) */
				// v_b_in = _mm256_add_epi64(v_z, v_x);
				// v_b_in = _mm256_mul_epu32(v_b_in, v_y2);
				for(l=0; l<4; l++) vb_in_64[l] = z[l+4] + vx64[l];
				for(l=0; l<4; l++) vb_in_64[l] = (uint32_t) vb_in_64[l] * v64_y2[l];
				// bernoulli_sampler_S1(b + 4, v_b_in, r1 + BASE_TABLE_SIZE);
				bernoulli_sampler_S1_64(b + 4, vb_in_64, r1 + BASE_TABLE_SIZE);
				// print256_num(v_b_in);
    			// printf("%.16lx %.16lx %.16lx %.16lx\n", vb_in_64[3], vb_in_64[2], vb_in_64[1], vb_in_64[0]);
			// if(j==1) for(l=0; l<8; l++) {printf("%.16lx ", b[l]);} printf("\n");
			// if(j==16) for(l=0; l<8; l++) {printf("%.16lx ", b64[l]);} printf("\n");	
				// for(l=0; l<8; l++) if(b[l]!=b64[l]) printf("wrong at %u b%u: %.16lx %.16lx\n", j, l, b[l], b64[l]);
				i = 0;
			}

			k = (r[2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE) + UNIFORM_REJ * UNIFORM_SIZE] >> i) & 0x1;
			i++;
			// printf("*%u %u %lu\n", j, i, (1 ^ ((b[i - 1]))));				
		} while (1 ^ ((b[i - 1] & ((z[i - 1] | -z[i - 1]) | (k | -k))) >> 63)); /* rejection condition: b=0 or ((b=1) && (z=0) && (k=0)) */

		mod=z[i-1];

		/*
		if(mod>10*225){
			printf("\n----LARGE----\n");
		}
		*/

		mod1=mod_prime(mod, 0);
		mod2=mod_prime(mod, 1);
		mod3=mod_prime(mod, 2);		
		
		sample[0][j]=(1-k)*mod1+k*mod_prime(SIFE_MOD_Q_I[0]-mod1, 0);
		sample[1][j]=(1-k)*mod2+k*mod_prime(SIFE_MOD_Q_I[1]-mod2, 1);
		sample[2][j]=(1-k)*mod3+k*mod_prime(SIFE_MOD_Q_I[2]-mod3, 2);


		#if SEC_LEVEL==2
			mod4=mod_prime(mod, 3);	
			sample[3][j]=(1-k)*mod4+k*mod_prime(SIFE_MOD_Q_I[3]-mod4, 3);
		#endif

		j++;

	}

}

//void gaussian_sampler_S2(unsigned char *seed, uint32_t sample[SIFE_NMODULI][SIFE_N], uint32_t slen)
void gaussian_sampler_S2_ori(aes256ctr_ctx *state, uint32_t sample[SIFE_NMODULI][SIFE_N], uint32_t slen)
{
	__m256i v_x, v_y1, v_y2, v_z, v_b_in;
	uint64_t z[8] __attribute__ ((aligned (32)));
	uint64_t b[8] __attribute__ ((aligned (32)));
	
	//unsigned char r[2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE) + UNIFORM_REJ * UNIFORM_SIZE + 1] __attribute__ ((aligned (32)));
	unsigned char r[384];
	unsigned char *r1;
	
	uint32_t i = 8, j = 0;
	uint64_t k;
	//aes256ctr_ctx state;
	//aes256ctr_init(&state, seed, 0);
	
	const uint32_t AES_ROUNDS=3;

	uint64_t mod;
	uint32_t mod1, mod2, mod3;

	#if SEC_LEVEL==2
		uint32_t mod4;	
	#endif

	while (j < slen)
	{
		do
		{
			if (i == 8)
			{
				/* x<--D_sigma_0, y<--U([0,k-1]), z=kx+y */
				//fastrandombytes(r, 2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE) + UNIFORM_REJ * UNIFORM_SIZE + 1);
				//aes256ctr_squeezeblocks(r, AES_ROUNDS, &state);
				aes256ctr_squeezeblocks(r, AES_ROUNDS, state);
				
				uniform_sampler_S2(r + 2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE), &v_y1, &v_y2);
				
				r1 = r;
				v_x = cdt_sampler(r1);
				v_x = _mm256_mul_epu32(v_x, V_K_K_K_K_S2);
				v_z = _mm256_add_epi64(v_x, v_y1);
				_mm256_store_si256((__m256i *)(z), v_z);
				/* b<--Bernoulli(exp(-y(y+2kx)/2sigma_0^2)) */
				v_b_in = _mm256_add_epi64(v_z, v_x);
				v_b_in = _mm256_mul_epu32(v_b_in, v_y1);
				bernoulli_sampler_S2(b, v_b_in, r1 + BASE_TABLE_SIZE);
				
				r1 = r + (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE);
				v_x = cdt_sampler(r1);
				v_x = _mm256_mul_epu32(v_x, V_K_K_K_K_S2);
				v_z = _mm256_add_epi64(v_x, v_y2);
				_mm256_store_si256((__m256i *)(z + 4), v_z);
				/* b<--Bernoulli(exp(-y(y+2kx)/2sigma_0^2)) */
				v_b_in = _mm256_add_epi64(v_z, v_x);
				v_b_in = _mm256_mul_epu32(v_b_in, v_y2);
				bernoulli_sampler_S2(b + 4, v_b_in, r1 + BASE_TABLE_SIZE);
				// printf("i==0\n");
				i = 0;
			}
			
			k = (r[2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE) + UNIFORM_REJ * UNIFORM_SIZE] >> i) & 0x1;
			i++;			
			// printf("i: %u\n", i);
		} while (1 ^ ((b[i - 1] & ((z[i - 1] | -z[i - 1]) | (k | -k))) >> 63)); /* rejection condition: b=0 or ((b=1) && (z=0) && (k=0)) */
		// if(j==593) printf("-----%u: %u %lu %lu, %lu\n", j, sample[0][j], b[i - 1] , z[i - 1], k);
		mod=z[i-1];
		/*
		if(mod>10*258376412UL){
			printf("\n----LARGE----\n");
		}
		*/

		mod1=mod_prime(mod, 0);
		mod2=mod_prime(mod, 1);
		mod3=mod_prime(mod, 2);		

		sample[0][j]=(1-k)*mod1+k*mod_prime(SIFE_MOD_Q_I[0]-mod1, 0);
		sample[1][j]=(1-k)*mod2+k*mod_prime(SIFE_MOD_Q_I[1]-mod2, 1);
		sample[2][j]=(1-k)*mod3+k*mod_prime(SIFE_MOD_Q_I[2]-mod3, 2);

		#if SEC_LEVEL==2
			mod4=mod_prime(mod, 3);	
			sample[3][j]=(1-k)*mod4+k*mod_prime(SIFE_MOD_Q_I[3]-mod4, 3);
		#endif

		j++;
	}
}

void gaussian_sampler_S2(aes256ctr_ctx *state, uint32_t sample[SIFE_NMODULI][SIFE_N], uint32_t slen)
{
	__m256i v_x, v_y1, v_y2, v_z, v_b_in;
	uint64_t z[8] __attribute__ ((aligned (32)));
	uint64_t b[8] __attribute__ ((aligned (32)));
	uint64_t b64[8] __attribute__ ((aligned (32)));
	uint64_t v64_y1[8] __attribute__ ((aligned (32)));
	uint64_t v64_y2[8] __attribute__ ((aligned (32)));
	uint64_t vx64[4] ={0};		uint64_t vb_in_64[4] ={0};	

	//unsigned char r[2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE) + UNIFORM_REJ * UNIFORM_SIZE + 1] __attribute__ ((aligned (32)));
	unsigned char r[384] = {0};
	unsigned char *r1;
	
	uint32_t i = 8, j = 0, l = 0;
	uint64_t k;
	//aes256ctr_ctx state;
	//aes256ctr_init(&state, seed, 0);
	
	const uint32_t AES_ROUNDS=3;

	uint64_t mod;
	uint32_t mod1, mod2, mod3;

	#if SEC_LEVEL==2
	uint32_t mod4;
	#endif

	while (j < slen)
	{
		do
		{			
			if (i == 8)
			{
				for(l=0; l<4; l++) vx64[l] = 0;
				/* x<--D_sigma_0, y<--U([0,k-1]), z=kx+y */
				//fastrandombytes(r, 2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE) + UNIFORM_REJ * UNIFORM_SIZE + 1);
				//aes256ctr_squeezeblocks(r, AES_ROUNDS, &state);
				aes256ctr_squeezeblocks(r, AES_ROUNDS, state);
				// for (k = 0; k < 384; k++) printf("%x ", r[k]);
				// uniform_sampler_S1(r + 2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE), &v_y1, &v_y2);
				uniform_sampler_S2_noAVX(r + 2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE), v64_y1, v64_y2);
  			
				r1 = r;
				cdt_sampler64(r1, vx64);				    	
				for(l=0; l<4; l++) vx64[l] = (uint32_t) vx64[l] * BINARY_SAMPLER_K_S2;							
				for(l=0; l<4; l++) z[l] = vx64[l] + v64_y1[l];					
				for(l=0; l<4; l++) vb_in_64[l] = z[l] + vx64[l];
				for(l=0; l<4; l++) vb_in_64[l] = (uint32_t) vb_in_64[l] * v64_y1[l];
				bernoulli_sampler_S2_64(b, vb_in_64, r1 + BASE_TABLE_SIZE);
	
				r1 = r + (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE);
				
				for(l=0; l<4; l++) vx64[l] = 0;
				cdt_sampler64(r1, vx64);	
				
				for(l=0; l<4; l++) vx64[l] = (uint32_t) vx64[l] * BINARY_SAMPLER_K_S2;						
				for(l=0; l<4; l++) z[l+4] = vx64[l] + v64_y2[l];
				// /* b<--Bernoulli(exp(-y(y+2kx)/2sigma_0^2)) */
				for(l=0; l<4; l++) vb_in_64[l] = z[l+4] + vx64[l];
				for(l=0; l<4; l++) vb_in_64[l] = (uint32_t) vb_in_64[l] * v64_y2[l];

				bernoulli_sampler_S2_64(b + 4, vb_in_64, r1 + BASE_TABLE_SIZE);
				i = 0;
			}

			k = (r[2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE) + UNIFORM_REJ * UNIFORM_SIZE] >> i) & 0x1;
			i++;					
			// if(j==88) {for (int mm = 0; mm < 8; ++mm) printf("%lu ", vb_in_64[i]); printf("\n");		}
			// printf("*%u %u %lu\n", j, i, (1 ^ ((b[i - 1]))));
			// if(j==81) printf("%u %u %lu %lu %lu\n", j, i, b[i - 1], z[i - 1], k);				
		} while (1 ^ ((b[i - 1] & ((z[i - 1] | -z[i - 1]) | (k | -k))) >> 63)); /* rejection condition: b=0 or ((b=1) && (z=0) && (k=0)) */

		mod=z[i-1];

		/*
		if(mod>10*225){
			printf("\n----LARGE----\n");
		}
		*/

		mod1=mod_prime(mod, 0);
		mod2=mod_prime(mod, 1);
		mod3=mod_prime(mod, 2);		
		
		sample[0][j]=(1-k)*mod1+k*mod_prime(SIFE_MOD_Q_I[0]-mod1, 0);
		sample[1][j]=(1-k)*mod2+k*mod_prime(SIFE_MOD_Q_I[1]-mod2, 1);
		sample[2][j]=(1-k)*mod3+k*mod_prime(SIFE_MOD_Q_I[2]-mod3, 2);


		#if SEC_LEVEL==2
			mod4=mod_prime(mod, 3);	
			sample[3][j]=(1-k)*mod4+k*mod_prime(SIFE_MOD_Q_I[3]-mod4, 3);
		#endif

		j++;

	}

}
//void gaussian_sampler_S3(unsigned char *seed, uint32_t sample[SIFE_NMODULI][SIFE_N], uint32_t slen)
void gaussian_sampler_S3(aes256ctr_ctx *state, uint32_t sample[SIFE_NMODULI][SIFE_N], uint32_t slen)
{
	__m256i v_x, v_y1, v_y2, v_z, v_b_in;
	uint64_t z[8] __attribute__ ((aligned (32)));
	uint64_t b[8] __attribute__ ((aligned (32)));
	
	//unsigned char r[2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE) + UNIFORM_REJ * UNIFORM_SIZE + 1] __attribute__ ((aligned (32)));
	unsigned char r[384];
	unsigned char *r1;
	
	uint32_t i = 8, j = 0;
	uint64_t k;
	//aes256ctr_ctx state;
	//aes256ctr_init(&state, seed, 0);
	
	const uint32_t AES_ROUNDS=3;

	uint64_t mod;
	uint32_t mod1, mod2, mod3;

	#if SEC_LEVEL==2
		uint32_t mod4;	
	#endif

	while (j < slen)
	{
		do
		{
			if (i == 8)
			{
				/* x<--D_sigma_0, y<--U([0,k-1]), z=kx+y */
				//fastrandombytes(r, 2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE) + UNIFORM_REJ * UNIFORM_SIZE + 1);
				//aes256ctr_squeezeblocks(r, AES_ROUNDS, &state);
				aes256ctr_squeezeblocks(r, AES_ROUNDS, state);
				
				uniform_sampler_S3(r + 2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE), &v_y1, &v_y2);
				
				r1 = r;
				v_x = cdt_sampler(r1);
				v_x = _mm256_mul_epu32(v_x, V_K_K_K_K_S3);
				v_z = _mm256_add_epi64(v_x, v_y1);
				_mm256_store_si256((__m256i *)(z), v_z);
				/* b<--Bernoulli(exp(-y(y+2kx)/2sigma_0^2)) */
				v_b_in = _mm256_add_epi64(v_z, v_x);
				v_b_in = _mm256_mul_epu32(v_b_in, v_y1);
				bernoulli_sampler_S3(b, v_b_in, r1 + BASE_TABLE_SIZE);
				
				r1 = r + (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE);
				v_x = cdt_sampler(r1);
				v_x = _mm256_mul_epu32(v_x, V_K_K_K_K_S3);
				v_z = _mm256_add_epi64(v_x, v_y2);
				_mm256_store_si256((__m256i *)(z + 4), v_z);
				/* b<--Bernoulli(exp(-y(y+2kx)/2sigma_0^2)) */
				v_b_in = _mm256_add_epi64(v_z, v_x);
				v_b_in = _mm256_mul_epu32(v_b_in, v_y2);
				bernoulli_sampler_S3(b + 4, v_b_in, r1 + BASE_TABLE_SIZE);

				i = 0;
			}
			
			k = (r[2 * (BASE_TABLE_SIZE + BERNOULLI_TABLE_SIZE) + UNIFORM_REJ * UNIFORM_SIZE] >> i) & 0x1;
			i++;			
		} while (1 ^ ((b[i - 1] & ((z[i - 1] | -z[i - 1]) | (k | -k))) >> 63)); /* rejection condition: b=0 or ((b=1) && (z=0) && (k=0)) */
		
		mod=z[i-1];
		/*
		if(z[i-1]>10UL*516752822UL){
			printf("\n----LARGE----\n");
		}
		*/

		mod1=mod_prime(mod, 0);
		mod2=mod_prime(mod, 1);
		mod3=mod_prime(mod, 2);		

		sample[0][j]=(1-k)*mod1+k*mod_prime(SIFE_MOD_Q_I[0]-mod1, 0);
		sample[1][j]=(1-k)*mod2+k*mod_prime(SIFE_MOD_Q_I[1]-mod2, 1);
		sample[2][j]=(1-k)*mod3+k*mod_prime(SIFE_MOD_Q_I[2]-mod3, 2);

		#if SEC_LEVEL==2
			mod4=mod_prime(mod, 3);	
			sample[3][j]=(1-k)*mod4+k*mod_prime(SIFE_MOD_Q_I[3]-mod4, 3);
		#endif

		j++;
	}
}

