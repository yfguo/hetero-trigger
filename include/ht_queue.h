#ifndef HT_QUEUE_H_INCLUDED
#define HT_QUEUE_H_INCLUDED

#include "ht_flag.h"

#include <cuda.h>
#include <cuda_runtime.h>

typedef struct HT_queue_op {
    volatile HT_flag_t *flag;
    struct HT_queue_op *prev;
    struct HT_queue_op *next;
    int id;
    void *data;
} HT_queue_op_t;

typedef void (*HT_queue_op_cb_t)(void * data);

enum {
    HT_QUEUE_OP_PRINTF = 0,
    MAX_NUM_HT_QUEUE_OP
} HT_queue_op_id_e;

extern HT_queue_op_cb_t queue_op_cbs[MAX_NUM_HT_QUEUE_OP];

int HT_queue_op_register(int id, HT_queue_op_cb_t cb);

int HT_queue_init();
int HT_queue_destroy();
int HT_queue_op_enqueue(int id, void *data, cudaStream_t stream);
int HT_queue_process();

int HT_printf_enqueue(uint64_t data, cudaStream_t stream);

#endif /* ifndef HT_QUEUE_H_INCLUDED */
