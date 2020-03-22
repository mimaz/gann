/*
 * dense-layer.c
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
#include "context.h"
#include "util.h"

#include <stdio.h>
#include <math.h>

struct dense_layer
{
    struct layer base;
    cl_program program;
    cl_kernel forward;
    cl_kernel derive_gradient;
    cl_kernel backward;
    cl_kernel backward_bias;
};

static void compile (struct layer *lay);
static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);

struct layer *
layer_make_dense (struct network *net,
                  int width, int height, int depth,
                  const char *activation)
{
    struct dense_layer *dense;
    struct layer *base;

    dense = g_new0 (struct dense_layer, 1);
    base = (struct layer *) dense;

    base->net = net;
    base->type = LAYER_DENSE;
    base->activation = activation;
    base->width = width;
    base->height = height;
    base->depth = depth;
    base->size = width * height * depth;
    base->compile = compile;
    base->forward = forward;
    base->backward = backward;
    base->release = release;

    network_push_layer (net, base);

    return base;
}

static void
compile (struct layer *lay)
{
    struct dense_layer *dense;
    struct context *ctx;
    g_autofree float *weight_v;
    GRand *rand;
    int i;

    g_assert (lay->type == LAYER_DENSE);
    g_assert ((lay->flags & LAYER_FLAG_COMPILED) == 0);

    dense = (struct dense_layer *) lay;
    ctx = lay->net->ctx;
    rand = lay->net->ctx->rand;

    lay->weights = lay->prev->size * lay->size;


    /*
     * Build buffers
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
     * Randomize weights
     */
    weight_v = g_new (float, lay->weights);

    for (i = 0; i < lay->weights; i++) {
        float r1 = 2.0f * (float) M_PI * (float) g_rand_double (rand);
        float r2 = -2.0f  * logf ((float) g_rand_double (rand));
        float d = (cosf (r1) * sqrtf (r2)) * sqrtf (2.0f / lay->prev->size);

        weight_v[i] = d;
    }

    clEnqueueWriteBuffer (ctx->queue,
                          lay->weight_mem,
                          CL_TRUE,
                          0, lay->weights * sizeof (cl_float),
                          weight_v,
                          0, NULL, NULL);


    /*
     * Clear buffers
     */
    context_clear_buffer (ctx, lay->bias_mem, lay->size, NULL);
    context_clear_buffer (ctx, lay->bias_delta_mem, lay->size, NULL);
    context_clear_buffer (ctx, lay->delta_mem, lay->weights, NULL);


    /*
     * Build program
     */
    context_program_clear (ctx);
    if (g_strcmp0 (lay->activation, "linear") != 0) {
        context_program_activation (ctx, lay->activation);
        context_program_option (ctx, "-DWITH_ACTIVATION");
    }
    context_program_option (ctx, "-DINPUTS=%d", lay->prev->size);
    context_program_option (ctx, "-DOUTPUTS=%d", lay->size);

    if (lay->net->flags & NETWORK_FLAG_BACKPROP) {
        context_program_option (ctx, "-DWITH_DERIVATIVE");
    }

    if (lay->prev->gradient_mem != 0) {
        context_program_option (ctx, "-DCALC_GRADIENT");
    }

    context_program_file (ctx, "dense-layer.cl");
    context_program_build (ctx, &dense->program);
    context_program_kernel (ctx, "forward", &dense->forward);
    context_program_kernel (ctx, "derive_gradient", &dense->derive_gradient);
    context_program_kernel (ctx, "backward", &dense->backward);
    context_program_kernel (ctx, "backward_bias", &dense->backward_bias);

    /*
     * Synchronize
     */
    clFinish (ctx->queue);

    /*
     * Mark layer compiled
     */
    lay->flags |= LAYER_FLAG_COMPILED;
}

static void
forward (struct layer *lay)
{
    struct dense_layer *dense;
    size_t globsiz, locsiz;
    cl_kernel kern;
    cl_int err;

    g_assert (lay->type == LAYER_DENSE);
    dense = (struct dense_layer *) lay;

    locsiz = MIN (lay->size, lay->net->ctx->group_size);
    globsiz = util_upper_multiply (lay->size, locsiz);
    kern = dense->forward;

    clSetKernelArg (kern, 0, sizeof (cl_mem), &lay->prev->value_mem);
    clSetKernelArg (kern, 1, sizeof (cl_mem), &lay->weight_mem);
    clSetKernelArg (kern, 2, sizeof (cl_mem), &lay->bias_mem);
    clSetKernelArg (kern, 3, sizeof (cl_mem), &lay->value_mem);

    if (lay->net->flags & NETWORK_FLAG_BACKPROP) {
        clSetKernelArg (kern, 4, sizeof (cl_mem), &lay->derivative_mem);
    }

    g_clear_pointer (&lay->forward_barrier, clReleaseEvent);

    err = clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                  kern, 1, NULL,
                                  &globsiz, &locsiz,
                                  UTIL_NONNULL (lay->prev->forward_barrier),
                                  UTIL_PTR_OR_NULL (lay->prev->forward_barrier),
                                  &lay->forward_barrier);
    g_assert (err == CL_SUCCESS);
}

static void
backward (struct layer *lay)
{
    struct dense_layer *dense;
    size_t globsiz, locsiz;
    float ratefactor;
    cl_event evderive, evbackprop, evbias, evlist[2];
    cl_kernel kern;
    cl_int err, evcount;


    g_assert (lay->type == LAYER_DENSE);
    g_assert (lay->gradient_mem != 0);


    evderive = NULL;
    evbackprop = NULL;
    evbias = NULL;

    ratefactor = lay->net->rate * (1 - lay->net->momentum) * lay->net->loss;
    dense = (struct dense_layer *) lay;


    /*
     * Apply derivative to current layer's gradients
     */
    locsiz = MIN (lay->size, lay->net->ctx->group_size);
    globsiz = ceilf ((float) lay->prev->size / locsiz) * locsiz;
    kern = dense->derive_gradient;

    clSetKernelArg (kern, 0, sizeof (cl_mem), &lay->derivative_mem);
    clSetKernelArg (kern, 1, sizeof (cl_mem), &lay->gradient_mem);

    err = clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                  kern, 1, NULL,
                                  &globsiz, &locsiz,
                                  0, NULL,
                                  &evderive);
    g_assert (err == CL_SUCCESS);



    /*
     * Create pending events list for weights backpropagation and
     * bias update tasks as they depend on derivative application
     * and next layer's barrier
     */
    evcount = 0;
    evlist[evcount++] = evderive;

    if (lay->next->backward_barrier != NULL) {
        evlist[evcount++] = lay->next->backward_barrier;
        clRetainEvent (lay->next->backward_barrier);
    }



    /*
     * Run weights backpropagation
     */
    locsiz = lay->net->ctx->group_size;
    globsiz = ceilf ((float) lay->prev->size / locsiz) * locsiz;
    kern = dense->backward;

    clSetKernelArg (kern, 0, sizeof (cl_mem), &lay->prev->value_mem);
    clSetKernelArg (kern, 1, sizeof (cl_mem), &lay->gradient_mem);
    clSetKernelArg (kern, 2, sizeof (cl_mem), &lay->prev->gradient_mem);
    clSetKernelArg (kern, 3, sizeof (cl_mem), &lay->weight_mem);
    clSetKernelArg (kern, 4, sizeof (cl_mem), &lay->delta_mem);
    clSetKernelArg (kern, 5, sizeof (cl_float), &ratefactor);
    clSetKernelArg (kern, 6, sizeof (cl_float), &lay->net->momentum);
    clSetKernelArg (kern, 7, sizeof (cl_float), &lay->net->decay);

    err = clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                  kern, 1, NULL,
                                  &globsiz, &locsiz,
                                  evcount, evlist,
                                  &evbackprop);
    g_assert (err == CL_SUCCESS);



    /*
     * Run bias update
     */
    globsiz = ceilf ((float) lay->size / locsiz) * locsiz;
    kern = dense->backward_bias;

    clSetKernelArg (kern, 0, sizeof (cl_mem), &lay->gradient_mem);
    clSetKernelArg (kern, 1, sizeof (cl_mem), &lay->bias_mem);
    clSetKernelArg (kern, 2, sizeof (cl_mem), &lay->bias_delta_mem);
    clSetKernelArg (kern, 3, sizeof (cl_float), &ratefactor);
    clSetKernelArg (kern, 4, sizeof (cl_float), &lay->net->momentum);
    clSetKernelArg (kern, 5, sizeof (cl_float), &lay->net->decay);

    err = clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                  kern, 1, NULL,
                                  &globsiz, &locsiz,
                                  evcount, evlist,
                                  &evbias);
    g_assert (err == CL_SUCCESS);



    /*
     * Release derive event already owned by both backpropagation tasks
     */
    clReleaseEvent (evderive);


    /*
     * Merge backpropagation events into one barrier
     */
    evlist[0] = evbackprop;
    evlist[1] = evbias;
    evcount = 2;

    g_clear_pointer (&lay->backward_barrier, clReleaseEvent);
    clEnqueueBarrierWithWaitList (lay->net->ctx->queue,
                                  evcount, evlist,
                                  &lay->backward_barrier);

    /*
     * And release backpropagation events already owned by the barrier
     */
    clReleaseEvent (evbackprop);
    clReleaseEvent (evbias);
}

static void
release (struct layer *lay)
{
    struct dense_layer *dense;

    g_assert (lay->type == LAYER_DENSE);
    dense = (struct dense_layer *) lay;

    g_clear_pointer (&lay->forward_barrier, clReleaseEvent);
    g_clear_pointer (&lay->backward_barrier, clReleaseEvent);

    clReleaseKernel (dense->forward);
    clReleaseKernel (dense->derive_gradient);
    clReleaseKernel (dense->backward);
    clReleaseKernel (dense->backward_bias);
    clReleaseProgram (dense->program);
    clReleaseMemObject (lay->value_mem);
    clReleaseMemObject (lay->derivative_mem);
    clReleaseMemObject (lay->gradient_mem);
    clReleaseMemObject (lay->bias_mem);
    clReleaseMemObject (lay->bias_delta_mem);
    clReleaseMemObject (lay->weight_mem);
    clReleaseMemObject (lay->delta_mem);
}
