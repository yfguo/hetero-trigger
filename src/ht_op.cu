#include <assert.h>
#include <stdio.h>
#include "ht_op.h"
#include "ht_flag.h"

int HT_stream_op_mode = HT_MODE_KERNEL;

#define CU_CALL(stmt) \
    do { \
        CUresult err = CUDA_SUCCESS; \
        err = (stmt); \
        if (err != CUDA_SUCCESS) { \
            cudaDeviceSynchronize(); \
            fprintf(stderr, "Failed at "#stmt" return %d\n", err); \
        } \
    } while (0)

#define CUDA_CALL(stmt) \
    do { \
        cudaError_t err = cudaSuccess; \
        err = (stmt); \
        if (err !=  cudaSuccess) { \
            cudaDeviceSynchronize(); \
            fprintf(stderr, "Failed at "#stmt" return %d\n", err); \
        } \
    } while (0)

__global__ void HT_kernel_set(volatile uint64_t* var, uint64_t val)
{
    *var = val;
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

void HT_set(volatile HT_flag_t* flag, uint64_t val, cudaStream_t stream)
{
    switch (HT_stream_op_mode) {
        case HT_MODE_HOST_FN:
            {
                host_fn_params *params = (host_fn_params *) malloc(sizeof(host_fn_params));
                params->flag = flag;
                params->val = val;
                CUDA_CALL(cudaLaunchHostFunc(stream, (cudaHostFn_t) HT_host_fn_set, params));
            }
            break;
        case HT_MODE_KERNEL:
            HT_kernel_set<<<1,1,0,stream>>>(flag->dev_ptr, val);
            CUDA_CALL(cudaPeekAtLastError());
            break;
        case HT_MODE_STREAM_MEM_OP:
            CU_CALL(cuStreamWriteValue64(stream, (CUdeviceptr) flag->dev_ptr, val, 0));
            break;
        default:
            assert(0);
    }
}

void HT_wait(volatile HT_flag_t* flag, uint64_t val, cudaStream_t stream)
{
    switch (HT_stream_op_mode) {
        case HT_MODE_HOST_FN:
            {
                host_fn_params *params = (host_fn_params *) malloc(sizeof(host_fn_params));
                params->flag = flag;
                params->val = val;
                CUDA_CALL(cudaLaunchHostFunc(stream, (cudaHostFn_t) HT_host_fn_wait, params));
            }
            break;
        case HT_MODE_KERNEL:
            HT_kernel_wait<<<1,1,0,stream>>>(flag->dev_ptr, val);
            CUDA_CALL(cudaPeekAtLastError());
            break;
        case HT_MODE_STREAM_MEM_OP:
            CU_CALL(cuStreamWaitValue64(stream, (CUdeviceptr) flag->dev_ptr, val,
                                        CU_STREAM_WAIT_VALUE_EQ));
            break;
        default:
            assert(0);
    }
}
