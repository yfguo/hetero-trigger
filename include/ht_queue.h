#ifndef HT_QUEUE_H_INCLUDED
#define HT_QUEUE_H_INCLUDED

#include "ht_flag.h"

#ifdef HAVE_CUDA
#include "ht_cuda.h"
#elif HAVE_HIP
#include "ht_hip.h"
#elif HAVE_ZE
#include "ht_ze.h"
#else
#error "No GPU support configured"
#endif

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
int HT_queue_op_enqueue(int id, void *data, HT_GPU_Stream_t stream);
int HT_queue_process();

int HT_printf_enqueue(uint64_t data, HT_GPU_Stream_t stream);

#endif /* ifndef HT_QUEUE_H_INCLUDED */
