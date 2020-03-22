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
    int kwidth;
    int kheight;
    int kstride;
    int kxshift;
    int kyshift;
    float *kbuffer;
    cl_program program;
    cl_kernel forward;
    cl_mem zero_mem;
};

static void compile (struct layer *lay);
static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);

struct layer *
layer_make_conv (struct network *net,
                 int size, int stride, int filters,
                 const char *activation)
{
    struct conv_layer *conv;
    struct layer *lay;

    conv = g_new0 (struct conv_layer, 1);
    lay = (struct layer *) conv;

    lay->net = net;
    lay->type = LAYER_CONV;
    lay->activation = activation;
    lay->depth = filters;
    lay->compile = compile;
    lay->forward = forward;
    lay->backward = backward;
    lay->release = release;

    conv->kwidth = size;
    conv->kheight = size;
    conv->kstride = stride;
    conv->kxshift = -size / 2;
    conv->kyshift = -size / 2;
    conv->kbuffer = g_new (float, lay->weights);

    network_push_layer (net, lay);

    return lay;
}

static void
compile (struct layer *lay)
{
    struct conv_layer *conv;
    struct layer *prev;
    struct context *ctx;
    g_autofree float *weight_v;
    int z, y, x, d, i;

    g_assert (lay->type == LAYER_CONV);

    conv = (struct conv_layer *) lay;
    ctx = lay->net->ctx;
    lay->weights = conv->kwidth * conv->kheight
        * lay->prev->depth * lay->depth;

    prev = lay->prev;

    lay->width = prev->width;
    lay->height = prev->height;
    lay->size = lay->width * lay->height * lay->depth;

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
    layer_create_buffer (lay, &conv->zero_mem,
                         lay->prev->depth, CL_MEM_READ_WRITE);

    /*
     * Set weights
     */
    weight_v = g_new (float, lay->weights);

    for (z = 0; z < lay->depth; z++) {
        for (y = 0; y < conv->kheight; y++) {
            for (x = 0; x < conv->kwidth; x++) {
                for (d = 0; d < lay->prev->depth; d++) {
                    i = z * conv->kwidth * conv->kheight * lay->prev->depth
                        + y * conv->kwidth * lay->prev->depth
                        + x * lay->prev->depth
                        + d;
                    if (x == 1 && y == 1) {
                        weight_v[i] = 1.0f / 3;
                    } else {
                        weight_v[i] = 0.0f;
                    }
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
    context_clear_buffer (ctx, conv->zero_mem, lay->prev->depth, NULL);

    /*
     * Build CL program
     */
    context_program_clear (ctx);
    context_program_file (ctx, "conv-layer.cl");
    context_program_option (ctx, "-DKERNEL_WIDTH=%d", conv->kwidth);
    context_program_option (ctx, "-DKERNEL_HEIGHT=%d", conv->kheight);
    context_program_option (ctx, "-DKERNEL_DEPTH=%d", lay->depth);
    context_program_option (ctx, "-DKERNEL_STRIDE=%d", conv->kstride);
    context_program_option (ctx, "-DKERNEL_X_SHIFT=%d", conv->kxshift);
    context_program_option (ctx, "-DKERNEL_Y_SHIFT=%d", conv->kyshift);
    context_program_option (ctx, "-DWIDTH=%d", lay->width);
    context_program_option (ctx, "-DHEIGHT=%d", lay->height);
    context_program_option (ctx, "-DDEPTH=%d", lay->depth);
    context_program_option (ctx, "-DINPUT_WIDTH=%d", lay->prev->width);
    context_program_option (ctx, "-DINPUT_HEIGHT=%d", lay->prev->height);
    context_program_option (ctx, "-DINPUT_DEPTH=%d", lay->prev->depth);
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
    size_t globsiz[3], locsiz[3];
    cl_kernel kern;
    cl_int err;

    g_assert (lay->type == LAYER_CONV);
    conv = (struct conv_layer *) lay;

    locsiz[0] = 1;
    locsiz[1] = 1;
    locsiz[2] = 1;
    globsiz[0] = lay->width;
    globsiz[1] = lay->height;
    globsiz[2] = lay->depth;

    kern = conv->forward;

    clSetKernelArg (kern, 0, sizeof (cl_mem), &lay->prev->value_mem);
    clSetKernelArg (kern, 1, sizeof (cl_mem), &lay->weight_mem);
    clSetKernelArg (kern, 2, sizeof (cl_mem), &conv->zero_mem);
    clSetKernelArg (kern, 3, sizeof (cl_mem), &lay->value_mem);

    g_clear_pointer (&lay->forward_barrier, clReleaseEvent);

    err = clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                  kern, 3, NULL,
                                  globsiz, locsiz,
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
    struct conv_layer *conv;

    g_assert (lay->type == LAYER_CONV);
    conv = (struct conv_layer *) lay;

    g_free (conv->kbuffer);
}
