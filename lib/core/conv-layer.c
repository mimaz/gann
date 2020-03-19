/*
 * conv-layer.c
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

struct conv_layer
{
    struct layer base;
    int size;
    int depth;
    int stride;
    cl_program program;
    cl_kernel forward;
};

static void compile (struct layer *lay);
static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);

struct layer *
layer_make_conv (struct network *net,
                 int size, int stride, int filters,
                 const char *activation,
                 struct layer *prev)
{
    struct conv_layer *conv;
    struct layer *lay;
    int width, height;

    conv = g_new0 (struct conv_layer, 1);
    lay = (struct layer *) conv;

    if (prev == NULL) {
        prev = network_layer_last (net);
    }

    prev->next = lay;

    width = prev->width;
    height = prev->height;

    lay->net = net;
    lay->prev = prev;
    lay->type = LAYER_CONV;
    lay->activation = activation;
    lay->width = width;
    lay->height = height;
    lay->depth = filters;
    lay->size = width * height * filters;
    lay->compile = compile;
    lay->forward = forward;
    lay->backward = backward;
    lay->release = release;

    conv->size = size;
    conv->depth = prev->depth;
    conv->stride = stride;

    network_push_layer (net, lay);

    return lay;
}

static void
compile (struct layer *lay)
{
    struct conv_layer *conv;
    struct context *ctx;
    g_autofree float *weight_v;
    int f, y, x, d, o;

    g_assert (lay->type == LAYER_CONV);

    conv = (struct conv_layer *) lay;
    ctx = lay->net->ctx;
    lay->weights = conv->size * conv->size * conv->depth * lay->depth;

    /*
     * Create buffers
     */
    layer_create_buffer (lay, &lay->value_mem,
                         lay->size, CL_MEM_READ_WRITE);
    layer_create_buffer (lay, &lay->derivative_mem,
                         lay->size, CL_MEM_READ_WRITE);
    layer_create_buffer (lay, &lay->gradient_mem,
                         lay->size, CL_MEM_READ_WRITE);
    layer_create_buffer (lay, &lay->bias_mem,
                         lay->size, CL_MEM_READ_WRITE);
    layer_create_buffer (lay, &lay->bias_delta_mem,
                         lay->size, CL_MEM_READ_WRITE);
    layer_create_buffer (lay, &lay->weight_mem,
                         lay->weights, CL_MEM_READ_WRITE);
    layer_create_buffer (lay, &lay->delta_mem,
                         lay->weights, CL_MEM_READ_WRITE);

    /*
     * Set weights
     */
    weight_v = g_new (float, lay->weights);

    for (f = 0; f < lay->depth; f++) {
        for (y = 0; y < conv->size; y++) {
            for (x = 0; x < conv->size; x++) {
                for (d = 0; d < conv->size; d++) {
                    o = f * conv->size * conv->size * conv->depth
                        + y * conv->size * conv->depth
                        + x * conv->depth
                        + d;
                    weight_v[o] = x;
                }
            }
        }
    }

    clEnqueueWriteBuffer (ctx->queue,
                          lay->weight_mem,
                          CL_TRUE,
                          0, lay->weights * sizeof (cl_float),
                          weight_v,
                          0, NULL, NULL);

    /*
     * Clear other buffers
     */
    context_clear_buffer (ctx, lay->bias_mem, lay->size, NULL);
    context_clear_buffer (ctx, lay->bias_delta_mem, lay->size, NULL);
    context_clear_buffer (ctx, lay->delta_mem, lay->weights, NULL);

    /*
     * Build CL program
     */
    context_program_clear (ctx);
    context_program_file (ctx, "conv-layer.cl");
    context_program_option (ctx, "-DSIZE=%d", conv->size);
    context_program_option (ctx, "-DSTRIDE=%d", conv->stride);
    context_program_option (ctx, "-DFILTERS=%d", lay->depth);
    context_program_option (ctx, "-DWIDTH=%d", lay->width);
    context_program_option (ctx, "-DHEIGHT=%d", lay->height);
    context_program_option (ctx, "-DDEPTH=%d", lay->prev->depth);
    context_program_build (ctx, &conv->program);
    context_program_kernel (ctx, "forward", &conv->forward);

    /*
     * Synchronize
     */
    clFinish (ctx->queue);

    /*
     * Mark compiled
     */
    lay->flags |= LAYER_FLAG_COMPILED;
}

static void
forward (struct layer *lay)
{
    struct conv_layer *conv;
    size_t globsiz, locsiz;
    cl_kernel kern;
    cl_int err;

    g_assert (lay->type == LAYER_CONV);
    conv = (struct conv_layer *) lay;

    locsiz = 1;
    globsiz = lay->size;

    kern = conv->forward;

    clSetKernelArg (kern, 0, sizeof (cl_mem), &lay->prev->value_mem);
    clSetKernelArg (kern, 1, sizeof (cl_mem), &lay->weight_mem);
    clSetKernelArg (kern, 2, sizeof (cl_mem), &lay->value_mem);

    g_clear_pointer (&lay->forward_barrier, clReleaseEvent);

    err = clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                  kern, 1, NULL,
                                  &globsiz, &locsiz,
                                  UTIL_NONNULL (lay->prev->forward_barrier),
                                  UTIL_PTR_OR_NULL (lay->prev->forward_barrier),
                                  &lay->forward_barrier);
    g_assert (err == CL_SUCCESS);
    clFinish (lay->net->ctx->queue);
}

static void
backward (struct layer *lay)
{
}

static void
release (struct layer *lay)
{
}
