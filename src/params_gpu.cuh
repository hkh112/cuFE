
#include <stdint.h>
// GPU settings
#define THREAD 512	// block size for gaussian sampler
#define LEN_THREAD 8

__device__  static uint32_t SIFE_MOD_Q_I_GPU[SIFE_NMODULI] = {16760833, 2147352577, 2130706433};//*
__device__ static const uint64_t SIFE_SCALE_M_MOD_Q_I_GPU[SIFE_NMODULI]={13798054, 441557681, 1912932552};	//*
