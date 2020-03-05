#include "layer.h"
#include "network.h"
#include "context.h"

#include <stdio.h>

#define USE_OPENCL

struct fully_layer
{
    struct layer base;
};

static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);

static cl_program
build_program (struct network *net,
               int inputs, int outputs)
{
    g_autofree char *opts;

    opts = g_strdup_printf ("-DINPUTS=%d "
                            "-DOUTPUTS=%d ",
                            inputs,
                            outputs);

    return context_build_program (net->ctx, opts,
                                  "fully-layer.cl",
                                  NULL);
}

static cl_mem
build_buffer (struct network *net,
              cl_uint flags,
              int value_count)
{
    cl_int err;
    cl_mem mem;

    mem = clCreateBuffer (net->ctx->context,
                          flags, value_count * sizeof (cl_float),
                          NULL, &err);
    g_assert (err == CL_SUCCESS);
    return mem;
}

struct layer *
layer_make_full (struct network *net,
                 enum activation_type activation,
                 int width, int height, int depth)
{
    struct layer *base, *prev;
    int size, weights, i;

    g_assert (sizeof (cl_float) == sizeof (gfloat));

    base = g_new0 (struct layer, 1);
    prev = network_layer_last (net);

    size = width * height * depth;
    weights = prev->size * size;

    base->net = net;
    base->prev = prev;
    base->type = LAYER_FULLY;
    base->activation = activation;
    base->value_mem = build_buffer (net, CL_MEM_READ_WRITE, size);
    base->gradient_mem = build_buffer (net, CL_MEM_READ_WRITE, size);
    base->bias_mem = build_buffer (net, CL_MEM_READ_WRITE, size);
    base->weight_mem = build_buffer (net, CL_MEM_READ_WRITE, weights);
    base->delta_mem = build_buffer (net, CL_MEM_READ_WRITE, weights);
    base->bias_delta_mem = build_buffer (net, CL_MEM_READ_WRITE, weights);
    base->program = build_program (net, base->prev->size, size);
    base->width = width;
    base->height = height;
    base->depth = depth;
    base->size = size;
    base->weights = weights;
    base->forward = forward;
    base->backward = backward;
    base->release = release;

    layer_create_kernel (base, KERNEL_REDUCE_INPUTS, "reduce_inputs");
    layer_create_kernel (base, KERNEL_BIAS_ACTIVATE, "bias_activate");
    layer_create_kernel (base, KERNEL_CLEAR_INPUT_GRADIENT, "clear_input_gradient");
    layer_create_kernel (base, KERNEL_BIAS_BACKPROP, "bias_backprop");
    layer_create_kernel (base, KERNEL_WEIGHT_BACKPROP, "weight_backprop");
    layer_create_kernel (base, KERNEL_DERIVE_GRADIENT, "derive_gradient");

    network_push_layer (net, base);

    g_autofree float *bias_v = g_new (float, size);
    g_autofree float *bias_delta_v = g_new (float, size);
    g_autofree float *delta_v = g_new (float, weights);
    g_autofree float *weight_v = g_new (float, weights);

    for (i = 0; i < weights; i++) {
        delta_v[i] = 0;
        weight_v[i] = (float) rand () / RAND_MAX / prev->size;
    }

    for (i = 0; i < size; i++) {
        bias_v[i] = 0;
        bias_delta_v[i] = 0;
    }

    clEnqueueWriteBuffer (net->ctx->queue,
                          base->bias_mem,
                          CL_TRUE,
                          0, size * sizeof (cl_float),
                          bias_v,
                          0, NULL, NULL);
    clEnqueueWriteBuffer (net->ctx->queue,
                          base->bias_delta_mem,
                          CL_TRUE,
                          0, size * sizeof (cl_float),
                          bias_delta_v,
                          0, NULL, NULL);
    clEnqueueWriteBuffer (net->ctx->queue,
                          base->delta_mem,
                          CL_TRUE,
                          0, weights * sizeof (cl_float),
                          delta_v,
                          0, NULL, NULL);
    clEnqueueWriteBuffer (net->ctx->queue,
                          base->weight_mem,
                          CL_TRUE,
                          0, weights * sizeof (cl_float),
                          weight_v,
                          0, NULL, NULL);

    return base;
}

static void
forward (struct layer *lay)
{
    cl_int err = CL_SUCCESS;
    size_t global_size, local_size;

    local_size = lay->prev->size;
    global_size = lay->size * local_size;
    g_assert (err == CL_SUCCESS);

    err = clSetKernelArg (lay->kernels[KERNEL_REDUCE_INPUTS], 0,
                          sizeof (cl_mem), &lay->prev->value_mem);
    err |= clSetKernelArg (lay->kernels[KERNEL_REDUCE_INPUTS], 1,
                           sizeof (cl_mem), &lay->weight_mem);
    err |= clSetKernelArg (lay->kernels[KERNEL_REDUCE_INPUTS], 2,
                           sizeof (cl_mem), &lay->value_mem);
    err |= clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                   lay->kernels[KERNEL_REDUCE_INPUTS],
                                   1, NULL,
                                   &global_size, &local_size,
                                   0, NULL, NULL);
    g_assert (err == CL_SUCCESS);
    local_size = 256;
    global_size = ceil((float) lay->size / local_size) * local_size;
    err = clSetKernelArg (lay->kernels[KERNEL_BIAS_ACTIVATE], 0,
                          sizeof (cl_mem), &lay->bias_mem);
    err |= clSetKernelArg (lay->kernels[KERNEL_BIAS_ACTIVATE], 1,
                           sizeof (cl_mem), &lay->value_mem);
    err |= clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                   lay->kernels[KERNEL_BIAS_ACTIVATE],
                                   1, NULL,
                                   &global_size, &local_size,
                                   0, NULL, NULL);
    g_assert (err == CL_SUCCESS);
}

static void
backward (struct layer *lay)
{
    cl_int err;
    size_t gsize, lsize;

    lsize = 256;
    gsize = ceil ((float) lay->prev->size / lsize) * lsize;
    err = clSetKernelArg (lay->kernels[KERNEL_CLEAR_INPUT_GRADIENT], 0,
                          sizeof (cl_mem), &lay->prev->gradient_mem);
    err |= clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                   lay->kernels[KERNEL_CLEAR_INPUT_GRADIENT],
                                   1, NULL,
                                   &gsize, &lsize,
                                   0, NULL, NULL);
    g_assert (err == CL_SUCCESS);

    lsize = 256;
    gsize = ceil ((float) lay->size / lsize) * lsize;
    err = clSetKernelArg (lay->kernels[KERNEL_BIAS_BACKPROP], 0,
                          sizeof (cl_mem), &lay->gradient_mem);
    err = clSetKernelArg (lay->kernels[KERNEL_BIAS_BACKPROP], 1,
                          sizeof (cl_mem), &lay->bias_delta_mem);
    err = clSetKernelArg (lay->kernels[KERNEL_BIAS_BACKPROP], 2,
                          sizeof (cl_mem), &lay->bias_mem);
    err = clSetKernelArg (lay->kernels[KERNEL_BIAS_BACKPROP], 3,
                          sizeof (cl_float), &lay->net->rate);
    err = clSetKernelArg (lay->kernels[KERNEL_BIAS_BACKPROP], 4,
                          sizeof (cl_float), &lay->net->momentum);
    err = clSetKernelArg (lay->kernels[KERNEL_BIAS_BACKPROP], 5,
                          sizeof (cl_float), &lay->net->decay);
    err |= clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                   lay->kernels[KERNEL_BIAS_BACKPROP],
                                   1, NULL,
                                   &gsize, &lsize,
                                   0, NULL, NULL);
    g_assert (err == CL_SUCCESS);

    lsize = lay->size;
    gsize = lay->size * lay->prev->size;
    err = clSetKernelArg (lay->kernels[KERNEL_WEIGHT_BACKPROP], 0,
                          sizeof (cl_mem), &lay->gradient_mem);
    err |= clSetKernelArg (lay->kernels[KERNEL_WEIGHT_BACKPROP], 1,
                           sizeof (cl_mem), &lay->delta_mem);
    err |= clSetKernelArg (lay->kernels[KERNEL_WEIGHT_BACKPROP], 2,
                           sizeof (cl_mem), &lay->weight_mem);
    err |= clSetKernelArg (lay->kernels[KERNEL_WEIGHT_BACKPROP], 3,
                           sizeof (cl_mem), &lay->value_mem);
    err |= clSetKernelArg (lay->kernels[KERNEL_WEIGHT_BACKPROP], 4,
                           sizeof (cl_mem), &lay->prev->gradient_mem);
    err |= clSetKernelArg (lay->kernels[KERNEL_WEIGHT_BACKPROP], 5,
                           sizeof (cl_mem), &lay->prev->value_mem);
    err |= clSetKernelArg (lay->kernels[KERNEL_WEIGHT_BACKPROP], 6,
                           sizeof (cl_float), &lay->net->rate);
    err |= clSetKernelArg (lay->kernels[KERNEL_WEIGHT_BACKPROP], 7,
                           sizeof (cl_float), &lay->net->momentum);
    err |= clSetKernelArg (lay->kernels[KERNEL_WEIGHT_BACKPROP], 8,
                           sizeof (cl_float), &lay->net->decay);
    err |= clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                   lay->kernels[KERNEL_WEIGHT_BACKPROP],
                                   1, NULL,
                                   &gsize, &lsize,
                                   0, NULL, NULL);
    g_assert (err == CL_SUCCESS);

    lsize = 256;
    gsize = ceil((float) lay->prev->size / 256) * 256;

    err = clSetKernelArg (lay->kernels[KERNEL_DERIVE_GRADIENT], 0,
                          sizeof (cl_mem), &lay->prev->value_mem);
    err |= clSetKernelArg (lay->kernels[KERNEL_DERIVE_GRADIENT], 1,
                          sizeof (cl_mem), &lay->prev->gradient_mem);
    err |= clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                   lay->kernels[KERNEL_DERIVE_GRADIENT],
                                   1, NULL,
                                   &gsize, &lsize,
                                   0, NULL, NULL);
    g_assert (err == CL_SUCCESS);
}

static void
release (struct layer *lay)
{
}
