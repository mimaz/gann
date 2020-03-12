/*
 * output-layer.c
 *
 * Copyright 2020 Mieszko Mazurek <mimaz@gmx.com>
 *
 * This file is part of Gann.
 *
 * Gann is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Gann is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Gann.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "layer.h"
#include "network.h"
#include "util.h"

struct output_layer
{
    struct layer base;
    cl_mem truth_mem;
    cl_mem loss_mem;
    cl_event truth_event;
    cl_program program;
    cl_kernel backprop_kern;
    float loss;
};

static void compile (struct layer *lay);
static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);

struct layer *
layer_make_output (struct network *net)
{
    struct output_layer *out;
    struct layer *base, *prev;
    int size;

    out = g_new0 (struct output_layer, 1);
    base = (struct layer *) out;
    prev = network_layer_last (net);
    size = prev->width * prev->height * prev->depth;

    g_assert (size == prev->size);

    base->net = net;
    base->prev = prev;
    base->type = LAYER_OUTPUT;
    base->width = prev->width;
    base->height = prev->height;
    base->depth = prev->depth;
    base->size = size;
    base->weights = 0;
    base->compile = compile;
    base->forward = forward;
    base->backward = backward;
    base->release = release;

    network_push_layer (net, base);

    return base;
}

void
layer_output_set_truth (struct layer *lay,
                        const float *data,
                        int size)
{
    struct output_layer *out;

    g_assert (lay->type == LAYER_OUTPUT);
    g_assert (lay->size == size);

    out = (struct output_layer *) lay;

    g_clear_pointer (&out->truth_event, clReleaseEvent);

    clEnqueueWriteBuffer (lay->net->ctx->queue,
                          out->truth_mem,
                          CL_TRUE,
                          0, size * sizeof (cl_float),
                          data, 0, NULL,
                          &out->truth_event);
}

static void
compile (struct layer *lay)
{
    struct output_layer *out;
    struct context *ctx;

    out = (struct output_layer *) lay;
    ctx = lay->net->ctx;

    layer_create_buffer (lay, &out->truth_mem,
                         lay->size, CL_MEM_READ_ONLY);
    layer_create_buffer (lay, &out->loss_mem,
                         1, CL_MEM_WRITE_ONLY);

    context_program_clear (ctx);
    context_program_file (ctx, "output-layer.cl");
    context_program_option (ctx, "-DSIZE=%d", lay->size);
    context_program_option (ctx, "-DSIZE_P2U=%d",
                            util_upper_power_2 (lay->size));

    if (lay->prev->gradient_mem != 0) {
        context_program_option (ctx, "-DCALC_GRADIENT");
    }

    context_program_build (ctx, &out->program);
    context_program_kernel (ctx, "backprop", &out->backprop_kern);

    lay->flags |= LAYER_FLAG_COMPILED;
}

static void
forward (struct layer *lay)
{
    g_assert (lay->type == LAYER_OUTPUT);
    g_assert (lay->size == lay->prev->size);
    g_assert (lay->prev != NULL);

    lay->value_mem = lay->prev->value_mem;
    lay->forward_barrier = lay->prev->forward_barrier;
}

static void
backward (struct layer *lay)
{
    struct output_layer *out;
    size_t globsiz, locsiz;
    cl_event evlist[2];
    cl_kernel kern;
    cl_int err, evcount;

    g_assert (lay->type == LAYER_OUTPUT);
    g_assert (lay->size == lay->prev->size);

    out = (struct output_layer *) lay;
    kern = out->backprop_kern;

    evcount = 0;

    if (out->truth_event != NULL) {
        evlist[evcount++] = out->truth_event;
    }

    if (lay->forward_barrier != NULL) {
        evlist[evcount++] = lay->forward_barrier;
    }

    clSetKernelArg (kern, 0, sizeof (cl_mem), &out->truth_mem);
    clSetKernelArg (kern, 1, sizeof (cl_mem), &lay->value_mem);
    clSetKernelArg (kern, 2, sizeof (cl_mem), &lay->prev->gradient_mem);
    clSetKernelArg (kern, 3, sizeof (cl_mem), &out->loss_mem);

    g_clear_pointer (&lay->backward_barrier, clReleaseEvent);

    locsiz = lay->size;
    globsiz = locsiz;
    err = clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                  kern, 1, NULL,
                                  &globsiz, &locsiz,
                                  evcount, evlist,
                                  &lay->backward_barrier);
    g_assert (err == CL_SUCCESS);

    clEnqueueReadBuffer (lay->net->ctx->queue,
                         out->loss_mem,
                         CL_TRUE,
                         0, sizeof (cl_float),
                         &out->loss,
                         evcount, evlist, NULL);

    /* TODO synchronize loss reading without blocking here */
    clFinish (lay->net->ctx->queue);
    lay->net->loss = out->loss;
}

static void
release (struct layer *lay)
{
    struct output_layer *out;

    g_assert (lay->type == LAYER_OUTPUT);
    out = (struct output_layer *) lay;

    g_clear_pointer (&lay->backward_barrier, clReleaseEvent);
    g_clear_pointer (&out->truth_event, clReleaseEvent);

    clReleaseKernel (out->backprop_kern);
    clReleaseProgram (out->program);
    clReleaseMemObject (out->truth_mem);
    clReleaseMemObject (out->loss_mem);
}
