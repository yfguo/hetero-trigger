/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef HT_FLAG_POOL_H_INCLUDED
#define HT_FLAG_POOL_H_INCLUDED

#include <stdint.h>

typedef void *HT_flag_pool_t;

typedef enum {
    HT_FLAG_POOL_WARN_POOL_SIZE = 1
} HT_flag_pool_status_e;

typedef struct {
    volatile uint64_t host_val;
    volatile uint64_t* dev_ptr;
} HT_flag_t;

typedef enum {
    HT_FLAG_QUEUED = 0,
    HT_FLAG_TRIGGERED,
    HT_FLAG_PENDDING
} HT_flag_state_e;

int HT_flag_pool_create_unsafe(intptr_t max_num_cells, HT_flag_pool_t * pool);
int HT_flag_pool_destroy_unsafe(HT_flag_pool_t pool);
int HT_flag_pool_alloc_flag(HT_flag_pool_t pool, volatile HT_flag_t **flag);
int HT_flag_pool_free_flag(HT_flag_pool_t pool, volatile HT_flag_t *flag);

#endif /* HT_FLAG_POOL_H_INCLUDED */
