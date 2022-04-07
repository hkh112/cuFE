
#include <stdint.h>
// GPU settings
#define THREAD 512	// block size for gaussian sampler
#define LEN_THREAD 8

#if SEC_LEVEL==0
__device__ static uint32_t SIFE_MOD_Q_I_GPU[SIFE_NMODULI] = {12289, 8257537, 536608769};
__device__ static const uint64_t SIFE_SCALE_M_MOD_Q_I_GPU[SIFE_NMODULI]={8654, 3309440, 415506400};	//*

#elif SEC_LEVEL==1
__device__  static uint32_t SIFE_MOD_Q_I_GPU[SIFE_NMODULI] = {16760833, 2147352577, 2130706433};//*
__device__ static const uint64_t SIFE_SCALE_M_MOD_Q_I_GPU[SIFE_NMODULI]={13798054, 441557681, 1912932552};	//*


#elif SEC_LEVEL==2
__device__  static uint32_t SIFE_MOD_Q_I_GPU[SIFE_NMODULI] = {114689, 1032193, 4293918721, 3221225473};//*
__device__ static const uint64_t SIFE_SCALE_M_MOD_Q_I_GPU[SIFE_NMODULI]={17902, 926438, 3160509796, 3131397206};	//*

#endif