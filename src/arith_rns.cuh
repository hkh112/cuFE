// #include "params.h"
#include "ntt_gpu.cuh"

#define k1_q1 24
#define k2_q1 14

#define k1_q2 31
#define k2_q2 17

#define k1_q3 31
#define k2_q3 24


#define k1_q1_minus_one ( (1UL<<k1_q1)-1 )
#define k1_q2_minus_one ( (1UL<<k1_q2)-1 )
#define k1_q3_minus_one ( (1UL<<k1_q3)-1 )



static __device__ uint32_t mod_prime_gpu(uint64_t m, uint32_t sel)
{
		while ( m > (2*(uint64_t)SIFE_MOD_Q_I_GPU[sel]) ) {
			if (sel == 0) {
				m = ( m& (k1_q1_minus_one) ) + ( ((m>>k1_q1)<<k2_q1) - (m>>k1_q1) );
			}
			else if (sel == 1) {
				m = ( m& (k1_q2_minus_one) ) + ( ((m>>k1_q2)<<k2_q2) - (m>>k1_q2) );
			}
			else{
				m = ( m& (k1_q3_minus_one) ) + ( ((m>>k1_q3)<<k2_q3) - (m>>k1_q3) );
			}
		}

	if (m >= SIFE_MOD_Q_I_GPU[sel]) {
		m = m - SIFE_MOD_Q_I_GPU[sel];
	}
	return (uint32_t)m;
}

static  __device__ uint32_t mul_mod_ntt_gpu(uint32_t a, uint32_t b, uint32_t sel)
{
	return (uint32_t)(mod_prime_gpu((uint64_t)a * (uint64_t)b, sel));
}


