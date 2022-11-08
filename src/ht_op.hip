#include <assert.h>
#include <stdio.h>
#include "ht_op.h"
#include "ht_flag.h"

int HT_stream_op_mode = HT_MODE_KERNEL;

#define HIP_CALL(stmt) \
    do { \
        hipError_t err = hipSuccess; \
        err = (stmt); \
        if (err !=  hipSuccess) { \
            hipDeviceSynchronize(); \
            fprintf(stderr, "Failed at "#stmt" return %d\n", err); \
        } \
    } while (0)

__global__ void HT_kernel_set(volatile uint64_t* var, uint64_t val)
{
    *var = val;
    __threadfence_system();
}

__global__ void HT_kernel_wait(volatile uint64_t* var, uint64_t val)
{
    while(*var != val);
}

typedef struct {
    volatile HT_flag_t *flag;
    uint64_t val;
} host_fn_params;

void HT_host_fn_set(host_fn_params *params)
{
    params->flag->host_val = params->val;
    free(params);
}

void HT_host_fn_wait(host_fn_params *params)
{
    while(params->flag->host_val != params->val);
    free(params);
}

void HT_set(volatile HT_flag_t* flag, uint64_t val, hipStream_t stream)
{
    switch (HT_stream_op_mode) {
        case HT_MODE_HOST_FN:
            {
                host_fn_params *params = (host_fn_params *) malloc(sizeof(host_fn_params));
                params->flag = flag;
                params->val = val;
                HIP_CALL(hipStreamAddCallback(stream, (hipStreamCallback_t) HT_host_fn_set, params));
            }
            break;
        case HT_MODE_KERNEL:
            HT_kernel_set<<<1,1,0,stream>>>(flag->dev_ptr, val);
            HIP_CALL(hipPeekAtLastError());
            break;
        case HT_MODE_STREAM_MEM_OP:
            HIP_CALL(hipStreamWriteValue64(stream, flag->dev_ptr, val, 0));
            break;
        default:
            assert(0);
    }
}

void HT_wait(volatile HT_flag_t* flag, uint64_t val, hipStream_t stream)
{
    switch (HT_stream_op_mode) {
        case HT_MODE_HOST_FN:
            {
                host_fn_params *params = (host_fn_params *) malloc(sizeof(host_fn_params));
                params->flag = flag;
                params->val = val;
                HIP_CALL(hipStreamAddCallback(stream, (hipStreamCallback_t) HT_host_fn_wait, params));
            }
            break;
        case HT_MODE_KERNEL:
            HT_kernel_wait<<<1,1,0,stream>>>(flag->dev_ptr, val);
            HIHIP_CALL(hipPeekAtLastError());
            break;
        case HT_MODE_STREAM_MEM_OP:
            HIP_CALL(hipStreamWaitValue64(stream, flag->dev_ptr, val,
                                          hipStreamWaitValueEq));
            break;
        default:
            assert(0);
    }
}
