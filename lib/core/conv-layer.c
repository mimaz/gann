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

struct conv_layer
{
    struct layer base;
    int size;
    int depth;
    int stride;
    cl_program program;
};

static void compile (struct layer *lay);
static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);

struct layer *
layer_make_conv (struct network *net,
                 int size, int depth, int stride,
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

    width = prev->width;
    height = prev->height;

    lay->net = net;
    lay->type = LAYER_CONV;
    lay->activation = activation;
    lay->width = width;
    lay->height = height;
    lay->depth = depth;
    lay->size = width * height * depth;
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
    context_program_build (ctx, &conv->program);

    /*
     * Synchronize
     */
    clFinish (ctx->queue);
}

static void
forward (struct layer *lay)
{
}

static void
backward (struct layer *lay)
{
}

static void
release (struct layer *lay)
{
}
