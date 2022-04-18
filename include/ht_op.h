#ifndef HT_STREAM_OP
#define HT_STREAM_OP

#include <stdint.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "ht_flag.h"

enum {
    HT_MODE_HOST_FN,
    HT_MODE_KERNEL,
    HT_MODE_STREAM_MEM_OP
};


__global__ void HT_kernel_set(volatile uint64_t* var, uint64_t val);
__global__ void HT_kernel_wait(volatile uint64_t* var, uint64_t val);

#ifdef __cplusplus
extern "C" {
#endif
    extern int HT_stream_op_mode;
    void HT_set(volatile HT_flag_t* flag, uint64_t val, cudaStream_t stream);
    void HT_wait(volatile HT_flag_t* flag, uint64_t val, cudaStream_t stream);
#ifdef __cplusplus
}
#endif

#endif
