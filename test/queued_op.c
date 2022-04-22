#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>
#include <pthread.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "ht_op.h"
#include "ht_flag.h"
#include "ht_queue.h"

#define MAX_NUM_FLAGS 100000
#define NS_TO_MS(t_ns) ((t_ns) / 1000 / 1000)
#define S_TO_MS(t_s) ((t_s) * 1000)
#define TIMESPEC_TO_MS(ts) ((double) S_TO_MS((ts).tv_sec) + (double) NS_TO_MS((ts).tv_nsec))

#define CU_CALL(stmt) \
    do { \
        CUresult err = CUDA_SUCCESS; \
        err = (stmt); \
        if (err != CUDA_SUCCESS) { \
            fprintf(stderr, "Failed at "#stmt" return %d\n", err); \
        } \
    } while (0)

#define CUDA_CALL(stmt) \
    do { \
        cudaError_t err = cudaSuccess; \
        err = (stmt); \
        if (err !=  cudaSuccess) { \
            fprintf(stderr, "Failed at "#stmt" return %d\n", err); \
        } \
    } while (0)

#define CUDA_CHECK_EVENT(ev) \
    do { \
        cudaError_t event_status; \
        event_status = cudaEventQuery(ev); \
        switch (event_status) { \
        case cudaSuccess: \
            printf("event success\n"); \
            break; \
        case cudaErrorNotReady: \
            printf("event not ready\n"); \
            break; \
        default: \
            printf("event error\n"); \
            break; \
        } \
    } while (0)

pthread_cond_t thread_running;

void *helper_thread(void *arg)
{
    while (1) {
        HT_queue_process();
    }
}

void usage(void)
{
    printf("set-perf: performance test for HT_set\n");
    printf("    ./set_perf -m <mode> -i <iteration> -n <num of ops>\n");
    printf("    mode: \n");
    printf("        1=MT_MODE_HOST_FN\n");
    printf("        2=MT_MODE_KERNEL\n");
    printf("        3=MT_MODE_STREAM_MEM_OP\n");
    printf("    num of ops: must be less than 1020\n");
}

int main(int argc, char *argv[])
{
    CUresult res;
    CUdevice dev;

    cudaStream_t stream;
    cudaEvent_t ev_start, ev_stop;

    struct timespec issue_start, issue_stop;
    struct timespec sync_start, sync_stop;

    HT_flag_pool_t flag_pool;
    volatile HT_flag_t *flag_sync;
    int n_flags = 1000;
    int n_iter = 1;

    pthread_t helper_th;

    if (argc != 7) {
        usage();
        exit(0);
    }

    n_flags = atoi(argv[6]);
    n_iter = atoi(argv[4]);
    int mode = atoi(argv[2]);
    switch (mode) {
        case 1:
            HT_stream_op_mode = HT_MODE_HOST_FN;
            break;
        case 2:
            HT_stream_op_mode = HT_MODE_KERNEL;
            break;
        case 3:
            HT_stream_op_mode = HT_MODE_STREAM_MEM_OP;
            break;
        default:
            fprintf(stderr, "Invalid mode %d\n", mode);
            exit(-1);
    }

    CU_CALL(cuInit(0));

    CUDA_CALL(cudaStreamCreate(&stream));
    CUDA_CALL(cudaEventCreate(&ev_start));
    CUDA_CALL(cudaEventCreate(&ev_stop));
    pthread_create(&helper_th, NULL, helper_thread, NULL);

    /* allocate variables */
    HT_flag_pool_create_unsafe(MAX_NUM_FLAGS, &flag_pool);
    HT_flag_pool_alloc_flag(flag_pool, &flag_sync);
    flag_sync->host_val = HT_FLAG_QUEUED;
    HT_queue_init();

    /* block start */
    HT_wait(flag_sync, HT_FLAG_TRIGGERED, stream);
    /* do work */
    cudaEventRecord(ev_start, stream);
    clock_gettime(CLOCK_REALTIME, &issue_start);
    for (int i = 0; i < n_flags; ++i) {
        HT_printf_enqueue(i, stream);
    }
    clock_gettime(CLOCK_REALTIME, &issue_stop);
    CUDA_CALL(cudaEventRecord(ev_stop, stream));
    /* all enqueue, start running stream */
    flag_sync->host_val = HT_FLAG_TRIGGERED;
    clock_gettime(CLOCK_REALTIME, &sync_start);
    cudaStreamSynchronize(stream);
    clock_gettime(CLOCK_REALTIME, &sync_stop);

    pthread_cancel(helper_th);
    /* end of work */

    float elapsed_time_ms = 0;
    CUDA_CALL(cudaEventElapsedTime(&elapsed_time_ms, ev_start, ev_stop));
    float issue_time_ms = 0;
    issue_time_ms = TIMESPEC_TO_MS(issue_stop) - TIMESPEC_TO_MS(issue_start);
    printf("Op issue/enqueue time for %d set: %f ms, avg %f ms\n",
           n_flags, issue_time_ms, issue_time_ms / n_flags);
    printf("Stream elapsed time for %d set kernel: %f ms, avg %f ms\n",
           n_flags, elapsed_time_ms, elapsed_time_ms / n_flags);
    printf("Stream sync time for %d set: %f ms\n",
           n_flags, TIMESPEC_TO_MS(sync_stop) - TIMESPEC_TO_MS(sync_stop));

    HT_flag_pool_free_flag(flag_pool, flag_sync);
    HT_flag_pool_destroy_unsafe(flag_pool);
    HT_queue_destroy();

    CUDA_CALL(cudaEventDestroy(ev_start));
    CUDA_CALL(cudaEventDestroy(ev_stop));
    CUDA_CALL(cudaStreamDestroy(stream));

    return 0;
}
