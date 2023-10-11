#ifndef HT_STREAM_OP
#define HT_STREAM_OP

#include <stdint.h>
#include "ht_flag.h"

enum {
    HT_MODE_HOST_FN,
    HT_MODE_KERNEL,
    HT_MODE_KERNEL_NOFENCE,
    HT_MODE_STREAM_MEM_OP
};

#ifdef HAVE_CUDA
#include "ht_cuda.h"
#elif HAVE_HIP
#include "ht_hip.h"
#elif HAVE_ZE
#include "ht_ze.h"
#else
#error "No GPU support configured"
#endif

__global__ void HT_kernel_set(volatile uint64_t* var, uint64_t val);
__global__ void HT_kernel_set_nofence(volatile uint64_t* var, uint64_t val);
__global__ void HT_kernel_wait(volatile uint64_t* var, uint64_t val);

#ifdef __cplusplus
extern "C" {
#endif
    extern int HT_stream_op_mode;
    void HT_set(volatile HT_flag_t* flag, uint64_t val, HT_GPU_Stream_t stream);
    void HT_wait(volatile HT_flag_t* flag, uint64_t val, HT_GPU_Stream_t stream);
#ifdef __cplusplus
}
#endif

#endif
