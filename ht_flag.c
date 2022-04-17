/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "ht_flag.h"
#include "utlist.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>

/* We reserve the front of each flag as flag header. It must be size of maximum alignment */
#define FLAG_HEADER_SIZE 16
#define FLAG_PAGE_SIZE 4096

#define FLAG_HEADER_TO_FLAG(p) (void *) ((char *) (p) + FLAG_HEADER_SIZE)
#define FLAG_TO_FLAG_HEADER(p) (flag_header_s *) ((char *) (p) - FLAG_HEADER_SIZE)

typedef struct flag_header flag_header_s;
typedef struct flag_block flag_block_s;

typedef void *(*HT_flag_pool_malloc_fn) (uintptr_t);
typedef void (*HT_flag_pool_free_fn) (void *);

struct flag_header {
    flag_block_s *block;
    flag_header_s *next;
};

struct flag_block {
    void *slab;
    intptr_t num_used_flags;
    flag_block_s *prev;
    flag_block_s *next;
    flag_header_s *free_list_head;
    flag_block_s *next_free;
};

typedef struct HT_flag_pool {
    intptr_t flag_size;
    intptr_t num_flags_in_block;
    intptr_t max_num_flags;

    HT_flag_pool_malloc_fn malloc_fn;
    HT_flag_pool_free_fn free_fn;

    intptr_t num_blocks;
    intptr_t max_num_blocks;
    flag_block_s *flag_blocks;
    flag_block_s *free_blocks_head;
    flag_block_s *free_blocks_tail;
} private_pool_s;

static int flag_block_alloc(private_pool_s * pool, flag_block_s ** block);

/* TODO: support for other GPU vendors */
static void* host_reg_malloc(uintptr_t size);
static void host_reg_free(void* ptr);

static void* host_reg_malloc(uintptr_t size)
{
    void* ptr = NULL;
    cudaError_t res = cudaSuccess;
    ptr = malloc(size);
    assert(ptr);
    cudaHostRegister(ptr, size, cudaHostRegisterPortable | cudaHostRegisterMapped);
    assert(res == cudaSuccess);
    return ptr;
}

static void host_reg_free(void* ptr)
{
    cudaHostUnregister(ptr);
    free(ptr);
}

int HT_flag_pool_create_unsafe(intptr_t max_num_flags, HT_flag_pool_t * pool)
{
    int rc = 0;
    private_pool_s *pool_obj;

    pool_obj = (private_pool_s *) malloc(sizeof(private_pool_s));

    /* internally enlarge the flag_size to accommodate flag header */
    pool_obj->flag_size = sizeof(HT_flag_t) + FLAG_HEADER_SIZE;
    /* Calculate based on page size. Avoids wastage in registered pages */
    pool_obj->num_flags_in_block = FLAG_PAGE_SIZE / pool_obj->flag_size;
    printf("num_flags_in_block %d\n", pool_obj->num_flags_in_block);
    assert(max_num_flags >= 0);
    if (max_num_flags == 0) {
        /* 0 means unlimited */
        pool_obj->max_num_blocks = 0;
    } else {
        pool_obj->max_num_blocks = max_num_flags / pool_obj->num_flags_in_block;
    }

    /* TODO: support for other GPU vendors */
    pool_obj->malloc_fn = host_reg_malloc;
    pool_obj->free_fn = host_reg_free;

    pool_obj->num_blocks = 0;

    pool_obj->flag_blocks = NULL;
    pool_obj->free_blocks_head = NULL;
    pool_obj->free_blocks_tail = NULL;

    *pool = (HT_flag_pool_t) pool_obj;

    return rc;
}

int HT_flag_pool_destroy_unsafe(HT_flag_pool_t pool)
{
    int rc = 0;
    private_pool_s *pool_obj = (private_pool_s *) pool;

    flag_block_s *curr, *tmp;
    DL_FOREACH_SAFE(pool_obj->flag_blocks, curr, tmp) {
        DL_DELETE(pool_obj->flag_blocks, curr);
        pool_obj->free_fn(curr->slab);
        free(curr);
    }

    /* free self */
    free(pool_obj);

    return rc;
}

static int flag_block_alloc(private_pool_s * pool, flag_block_s ** block)
{
    int rc = 0;
    flag_block_s *new_block = NULL;

    new_block = (flag_block_s *) malloc(sizeof(flag_block_s));

    if (!new_block) {
        rc = -1;
        goto fn_fail;
    }

    new_block->slab = pool->malloc_fn(pool->num_flags_in_block * pool->flag_size);
    if (!new_block->slab) {
        rc = -1;
        goto fn_fail;
    }

    new_block->free_list_head = NULL;
    /* init flag headers */
    for (int i = 0; i < pool->num_flags_in_block; i++) {
        flag_header_s *p = (void *) ((char *) new_block->slab + i * pool->flag_size);
        p->block = new_block;
        /* push to free list */
        p->next = new_block->free_list_head;
        new_block->free_list_head = p;
        /* init dev_ptr for flag */
        HT_flag_t *flag = (HT_flag_t *) FLAG_HEADER_TO_FLAG(p);
        cudaError_t ret = cudaSuccess;
        ret = cudaHostGetDevicePointer((void **) &flag->dev_ptr, (void *) &(flag->host_val), 0);
        assert(ret == cudaSuccess);
    }

    new_block->num_used_flags = 0;
    new_block->next = NULL;

    *block = new_block;

  fn_exit:
    return rc;
  fn_fail:
    if (new_block) {
        pool->free_fn(new_block->slab);
    }
    free(new_block);
    *block = NULL;
    goto fn_exit;
}

static void flag_block_free(private_pool_s * pool_obj, flag_block_s * block)
{
    pool_obj->free_fn(block->slab);
    free(block);
}

static void append_free_blocks(private_pool_s * pool_obj, flag_block_s * block)
{
    block->next_free = NULL;
    if (pool_obj->free_blocks_head == NULL) {
        pool_obj->free_blocks_head = block;
        pool_obj->free_blocks_tail = block;
    } else {
        pool_obj->free_blocks_tail->next_free = block;
        pool_obj->free_blocks_tail = block;
    }
}

static void shift_free_blocks(private_pool_s * pool_obj, flag_block_s * block)
{
    pool_obj->free_blocks_head = pool_obj->free_blocks_head->next_free;
    if (pool_obj->free_blocks_head == NULL) {
        pool_obj->free_blocks_tail = NULL;
    }
}

static void remove_free_blocks(private_pool_s * pool_obj, flag_block_s * block)
{
    if (pool_obj->free_blocks_head == block) {
        shift_free_blocks(pool_obj, block);
    } else {
        flag_block_s *tmp_block = pool_obj->free_blocks_head;
        while (tmp_block->next_free != block) {
            tmp_block = tmp_block->next_free;
        }
        assert(tmp_block->next_free == block);
        tmp_block->next_free = tmp_block->next_free->next_free;
        if (pool_obj->free_blocks_tail == block) {
            pool_obj->free_blocks_tail = tmp_block;
        }
    }
}

int HT_flag_pool_alloc_flag(HT_flag_pool_t pool, HT_flag_t **flag)
{
    int rc = 0;
    private_pool_s *pool_obj = (private_pool_s *) pool;

    if (!pool_obj->free_blocks_head) {
        /* try allocate more blocks if no free flag found */
        if (pool_obj->max_num_blocks > 0 && pool_obj->num_blocks >= pool_obj->max_num_blocks) {
            rc = -1;
            goto fn_fail;
        }

        flag_block_s *new_block;
        rc = flag_block_alloc(pool_obj, &new_block);
        if (rc != 0) {
            goto fn_fail;
        }

        pool_obj->num_blocks++;
        DL_APPEND(pool_obj->flag_blocks, new_block);
        append_free_blocks(pool_obj, new_block);
    }

    flag_block_s *block = NULL;
    flag_header_s *flag_h = NULL;

    block = pool_obj->free_blocks_head;
    flag_h = block->free_list_head;
    block->free_list_head = flag_h->next;

    *flag = FLAG_HEADER_TO_FLAG(flag_h);
    assert(flag_h->block == block);
    block->num_used_flags++;

    /* remove from free_blocks_head if all flags are used */
    if (block->num_used_flags == pool_obj->num_flags_in_block) {
        shift_free_blocks(pool_obj, block);
    }

  fn_exit:
    return rc;
  fn_fail:
    *flag = NULL;
    goto fn_exit;
}

int HT_flag_pool_free_flag(HT_flag_pool_t pool, HT_flag_t *flag)
{
    int rc = 0;
    private_pool_s *pool_obj = (private_pool_s *) pool;
    flag_block_s *block = NULL;
    flag_header_s *flag_h = NULL;

    if (flag == NULL) {
        goto fn_exit;
    }

    flag_h = FLAG_TO_FLAG_HEADER(flag);
    block = flag_h->block;

    flag_h->next = block->free_list_head;
    block->free_list_head = flag_h;

    block->num_used_flags--;

    if (block->num_used_flags == pool_obj->num_flags_in_block - 1) {
        append_free_blocks(pool_obj, block);
    } else if (block->num_used_flags == 0) {
        /* Avoid frequent re-allocation by preserving the last block in unlimited pool.
         * All blocks will be freed when the pool is destroyed */
        if (pool_obj->max_num_flags == 0 && pool_obj->num_blocks > 1) {
            remove_free_blocks(pool_obj, block);
            DL_DELETE(pool_obj->flag_blocks, block);
            flag_block_free(pool_obj, block);
            pool_obj->num_blocks--;
        }
    }

  fn_exit:
    return rc;
}
