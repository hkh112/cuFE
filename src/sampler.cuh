#include "params.h"
/*ceil(DESIRED_SIGMA/sigma_0) sigma_0=sqrt( 1/(2*log(2) ) )*/


#define BINARY_SAMPLER_K_S1 266		//sigma_1
#define BINARY_SAMPLER_K_S2 304214978LL	//sigma_2
#define BINARY_SAMPLER_K_S3 608429953LL	//sigma_3


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


__constant__ const uint64_t CDT[9][2] = {
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

/* coefficients of the exp evaluation polynomial */
static __constant__ uint64_t EXP_COFF64[10] = {0x3e833b70ffa2c5d4,
									0x3eb4a480fda7e6e1,
									0x3ef01b254493363f,
									0x3f242e0e0aa273cc,
									0x3f55d8a2334ed31b,
									0x3f83b2aa56db0f1a,
									0x3fac6b08e11fc57e,
									0x3fcebfbdff556072,
									0x3fe62e42fefa7fe6,
									0x3ff0000000000000};	

static const uint64_t V_RES_MANTISSA64 = 1LL << EXP_MANTISSA_PRECISION;
static const uint64_t V_RES_EXPONENT64 = (uint64_t)R_EXPONENT_L - 1023 + 1;


/* make sure that Pr(rerun the PRG)<=2^(-64) */
__device__ inline void uniform_sampler_S1_gpu(unsigned char *r, uint64_t* sam64y1, uint64_t* sam64y2)
{
	uint64_t sample[8];
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
	}

	for(i=0; i<4; i++ ) sam64y1[i] = sample[i];
	for(i=0; i<4; i++ ) sam64y2[i] = sample[i + 4];
}

__device__ inline void uniform_sampler_S2_gpu(unsigned char *r, uint64_t* sam64y1, uint64_t* sam64y2)
{
	uint64_t sample[8];
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
	}

	for(i=0; i<4; i++ ) sam64y1[i] = sample[i];
	for(i=0; i<4; i++ ) sam64y2[i] = sample[i + 4];
}

__device__ inline void uniform_sampler_S3_gpu(unsigned char *r, uint64_t* sam64y1, uint64_t* sam64y2)
{
	uint64_t sample[8];
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
	}

	for(i=0; i<4; i++ ) sam64y1[i] = sample[i];
	for(i=0; i<4; i++ ) sam64y2[i] = sample[i + 4];
}

static __device__ uint64_t GETU64(uint8_t *r)
{
	
	uint64_t temp = ((uint64_t)r[7] << 56) ^ ((uint64_t)r[6] << 48) ^ ((uint64_t)r[5] << 40)  ^ ((uint64_t)r[4] << 32) ^ ((uint64_t)r[3] << 24) ^ ((uint64_t)r[2] << 16)  ^ ((uint64_t)r[1] << 8) ^ (uint64_t)r[0] ;
	return temp;
}

/* constant time CDT sampler */
__device__ inline void cdt_sampler_gpu(uint8_t *r, uint64_t *x)
{
	uint64_t r1[4] = {0};	uint64_t r2[4] = {0};
	uint64_t r1_lt_cdt0[4] = {0};	uint64_t r2_lt_cdt1[4] = {0};
	uint64_t r2_eq_cdt1[4] = {0};
	uint64_t b[4] = {0};

	int i, j;
	
	for(i=0; i<4; i++) r1[i] = GETU64(r+i*8) ;

	for(i=0; i<4; i++) r2[i] = GETU64(r+i*8 + 32) ;
	for(i=0; i<4; i++)	r1[i] &= CDT_LOW_MASK;	
	for(i=0; i<4; i++)	r2[i] &= CDT_LOW_MASK;

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
	}		
}

__device__ inline void bernoulli_sampler_S1_gpu(uint64_t *b, uint64_t *x, unsigned char *r)
{	
	int i=0;
	double vx64[4] = {0}, vx1_64[4] = {0}, vx2_64[4] = {0}, vsum64[4] = {0};
	int64_t vt64[4] = {0}, vres64[4] = {0}, vres_mantissa64[4] = {0}, vres_exponent64[4] = {0}, vr_mantissa64[4] = {0}, vr_exponent64[4] = {0}, vr_exponent2_64[4] = {0}, vres_eq_164[4] = {0}, vr_lt_vres_mantissa64[4] = {0}, vr_lt_vres_exponent64[4] = {0};
	int64_t mask = 0x4330000000000000, *p;	

	/* 2^x=2^(floor(x)+a)=2^(floor(x))*2^a, where a is in [0,1]
	 * we only evaluate 2^a by using a polynomial */
	p = (int64_t*) &mask;		

	for(i=0; i<4; i++) x[i] |= *p;
	for(i=0; i<4; i++) vx64[i] = (double) (x[i] - 0x4330000000000000);
	for(i=0; i<4; i++) vx64[i] = vx64[i] * BINARY_SAMPLER_K_2_INV_S1;
	for(i=0; i<4; i++) vx1_64[i] = floor(vx64[i]);		
	for(i=0; i<4; i++) vx2_64[i] = vx1_64[i] + 0x0018000000000000;
	for(i=0; i<4; i++) vt64[i] = (int64_t) vx2_64[i] - 0x0018000000000000;
	for(i=0; i<4; i++) vt64[i] = vt64[i] <<52;
	/* evaluate 2^a */
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
	/* combine to compute 2^x */
	for(i=0; i<4; i++) {
		p = (int64_t*) &vsum64[i];
		vres64[i] = vt64[i] + *p;
	}

	/* compute the Bernoulli value */
	for(i=0; i<4; i++) vres_mantissa64[i] = vres64[i] & EXP_MANTISSA_MASK;
	for(i=0; i<4; i++) vres_mantissa64[i] |= V_RES_MANTISSA64;
	for(i=0; i<4; i++) vres_exponent64[i] = vres64[i] >> EXP_MANTISSA_PRECISION;
	for(i=0; i<4; i++) vres_exponent64[i] += V_RES_EXPONENT64;
	for(i=0; i<4; i++) vres_exponent64[i] = 1LL << vres_exponent64[i];		

	for(i=0; i<4; i++) vr_mantissa64[i] = GETU64(r+i*8) ;
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
}

__device__ inline void bernoulli_sampler_S2_gpu(uint64_t *b, uint64_t *x, unsigned char *r)
{	
	int i=0;
	double vx64[4] = {0}, vx1_64[4] = {0}, vx2_64[4] = {0}, vsum64[4] = {0};
	int64_t vt64[4] = {0}, vres64[4] = {0}, vres_mantissa64[4] = {0}, vres_exponent64[4] = {0}, vr_mantissa64[4] = {0}, vr_exponent64[4] = {0}, vr_exponent2_64[4] = {0}, vres_eq_164[4] = {0}, vr_lt_vres_mantissa64[4] = {0}, vr_lt_vres_exponent64[4] = {0};
	// Comment V_INT64_DOUBLE in Hex form
	int64_t mask = 0x4330000000000000, *p;
	double *p2;
	/* 2^x=2^(floor(x)+a)=2^(floor(x))*2^a, where a is in [0,1]
	 * we only evaluate 2^a by using a polynomial */
	p = (int64_t*) &mask;		

	for(i=0; i<4; i++) x[i] |= *p;
	for(i=0; i<4; i++) 
	{
		p2 = (double*) &x[i];
		vx64[i] = *p2 - 4503599627370496.0;
	}
	// for(i=0; i<4; i++) vx64[i] = (double) (x[i] - 0x4330000000000000);
	for(i=0; i<4; i++) vx64[i] = vx64[i] * BINARY_SAMPLER_K_2_INV_S2;
	for(i=0; i<4; i++) vx1_64[i] = floor(vx64[i]);		
	for(i=0; i<4; i++) vx2_64[i] = vx1_64[i] + 0x0018000000000000;
	for(i=0; i<4; i++) vt64[i] = (int64_t) vx2_64[i] - 0x0018000000000000;
	for(i=0; i<4; i++) vt64[i] = vt64[i] <<52;
	/* evaluate 2^a */
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
	/* combine to compute 2^x */
	for(i=0; i<4; i++) {
		p = (int64_t*) &vsum64[i];
		vres64[i] = vt64[i] + *p;
	}

	/* compute the Bernoulli value */
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
	//for(i=0; i<4; i++) vres_exponent64[i] = 1LL << vres_exponent64[i];		

	for(i=0; i<4; i++) vr_mantissa64[i] = GETU64(r+i*8) ;
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
}


__device__ inline void bernoulli_sampler_S3_gpu(uint64_t *b, uint64_t *x, unsigned char *r)
{	
	int i=0;
	double vx64[4] = {0}, vx1_64[4] = {0}, vx2_64[4] = {0}, vsum64[4] = {0};
	int64_t vt64[4] = {0}, vres64[4] = {0}, vres_mantissa64[4] = {0}, vres_exponent64[4] = {0}, vr_mantissa64[4] = {0}, vr_exponent64[4] = {0}, vr_exponent2_64[4] = {0}, vres_eq_164[4] = {0}, vr_lt_vres_mantissa64[4] = {0}, vr_lt_vres_exponent64[4] = {0};
	// Comment V_INT64_DOUBLE in Hex form
	int64_t mask = 0x4330000000000000, *p;
	double *p2;
	/* 2^x=2^(floor(x)+a)=2^(floor(x))*2^a, where a is in [0,1]
	 * we only evaluate 2^a by using a polynomial */
	p = (int64_t*) &mask;		

	for(i=0; i<4; i++) x[i] |= *p;
	for(i=0; i<4; i++) 
	{
		p2 = (double*) &x[i];
		vx64[i] = *p2 - 4503599627370496.0;
	}
	// for(i=0; i<4; i++) vx64[i] = (double) (x[i] - 0x4330000000000000);
	for(i=0; i<4; i++) vx64[i] = vx64[i] * BINARY_SAMPLER_K_2_INV_S3;
	for(i=0; i<4; i++) vx1_64[i] = floor(vx64[i]);		
	for(i=0; i<4; i++) vx2_64[i] = vx1_64[i] + 0x0018000000000000;
	for(i=0; i<4; i++) vt64[i] = (int64_t) vx2_64[i] - 0x0018000000000000;
	for(i=0; i<4; i++) vt64[i] = vt64[i] <<52;
	/* evaluate 2^a */
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
	/* combine to compute 2^x */
	for(i=0; i<4; i++) {
		p = (int64_t*) &vsum64[i];
		vres64[i] = vt64[i] + *p;
	}

	/* compute the Bernoulli value */
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
	//for(i=0; i<4; i++) vres_exponent64[i] = 1LL << vres_exponent64[i];		

	for(i=0; i<4; i++) vr_mantissa64[i] = GETU64(r+i*8) ;
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
}