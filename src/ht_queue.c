#include "ht_queue.h"
#include "ht_op.h"
#include "utlist.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <inttypes.h>

HT_queue_op_cb_t queue_op_cbs[MAX_NUM_HT_QUEUE_OP];

HT_flag_pool_t flag_pool;
pthread_mutex_t queue_lock;
HT_queue_op_t *queue;

typedef uint64_t printf_data;

void printf_cb(void *data)
{
    printf("data %"PRIu64"\n", (printf_data) data);
}

int HT_queue_init()
{
    HT_queue_op_register(HT_QUEUE_OP_PRINTF, printf_cb);
    HT_flag_pool_create_unsafe(10000, &flag_pool);
    pthread_mutex_init(&queue_lock, NULL);
    queue = NULL;
    return 0;
}

int HT_queue_destroy()
{
    HT_flag_pool_destroy_unsafe(flag_pool);
    pthread_mutex_destroy(&queue_lock);
    return 0;
}

int HT_queue_op_register(int id, HT_queue_op_cb_t cb)
{
    queue_op_cbs[id] = cb;
    return 0;
}

int HT_queue_op_enqueue(int id, void *data, cudaStream_t stream)
{
    HT_queue_op_t *q_op = NULL;
    q_op = (HT_queue_op_t *) malloc(sizeof(HT_queue_op_t));
    assert(q_op);
    HT_flag_pool_alloc_flag(flag_pool, &(q_op->flag));
    q_op->flag->host_val = HT_FLAG_QUEUED;
    q_op->prev = NULL;
    q_op->next = NULL;
    q_op->id = id;
    q_op->data = data;
    /* enqueue */
    pthread_mutex_lock(&queue_lock);
    DL_APPEND(queue, (HT_queue_op_t *) q_op);
    pthread_mutex_unlock(&queue_lock);
    /* enqueue to GPU stream */
    HT_set(q_op->flag, HT_FLAG_TRIGGERED, stream);
    return 0;
}

int HT_printf_enqueue(uint64_t data, cudaStream_t stream)
{
    return HT_queue_op_enqueue(HT_QUEUE_OP_PRINTF, (void *) data, stream);
}

int HT_queue_process()
{
    HT_queue_op_t *op = NULL, *tmp = NULL;
    pthread_mutex_lock(&queue_lock);
    DL_FOREACH_SAFE(queue, op, tmp) {
        if (op->flag->host_val == HT_FLAG_TRIGGERED) {
            queue_op_cbs[op->id](op->data);
            DL_DELETE(queue, op);
            free(op);
        }
    }
    pthread_mutex_unlock(&queue_lock);
    return 0;
}
