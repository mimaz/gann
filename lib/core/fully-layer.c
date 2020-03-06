#include "layer.h"
#include "network.h"
#include "context.h"

#include <stdio.h>

#define USE_OPENCL

struct fully_layer
{
    struct layer base;
    cl_kernel reduce_inputs;
    cl_kernel bias_activate;
    cl_kernel clear_input_gradient;
    cl_kernel bias_backprop;
    cl_kernel weight_backprop;
    cl_kernel derive_gradient;
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
    struct fully_layer *fully;
    struct layer *base, *prev;
    int size, weights, i;

    g_assert (sizeof (cl_float) == sizeof (gfloat));

    fully = g_new0 (struct fully_layer, 1);
    base = (struct layer *) fully;
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

    layer_create_kernel (base, &fully->reduce_inputs, "reduce_inputs");
    layer_create_kernel (base, &fully->bias_activate, "bias_activate");
    layer_create_kernel (base, &fully->clear_input_gradient, "clear_input_gradient");
    layer_create_kernel (base, &fully->bias_backprop, "bias_backprop");
    layer_create_kernel (base, &fully->weight_backprop, "weight_backprop");
    layer_create_kernel (base, &fully->derive_gradient, "derive_gradient");

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
    struct fully_layer *fully;
    cl_int err;
    size_t global_size, local_size;
    cl_kernel kern;

    g_assert (lay->type == LAYER_FULLY);

    fully = (struct fully_layer *) lay;

    local_size = lay->prev->size;
    global_size = lay->size * local_size;
    kern = fully->reduce_inputs;

    clSetKernelArg (kern, 0, sizeof (cl_mem), &lay->prev->value_mem);
    clSetKernelArg (kern, 1, sizeof (cl_mem), &lay->weight_mem);
    clSetKernelArg (kern, 2, sizeof (cl_mem), &lay->value_mem);
    err = clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                  kern, 1, NULL,
                                  &global_size, &local_size,
                                  0, NULL, NULL);
    g_assert (err == CL_SUCCESS);

    local_size = 256;
    global_size = ceil((float) lay->size / local_size) * local_size;
    kern = fully->bias_activate;

    clSetKernelArg (kern, 0, sizeof (cl_mem), &lay->bias_mem);
    clSetKernelArg (kern, 1, sizeof (cl_mem), &lay->value_mem);
    err = clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                  kern, 1, NULL,
                                  &global_size, &local_size,
                                  0, NULL, NULL);
    g_assert (err == CL_SUCCESS);
}

static void
backward (struct layer *lay)
{
    struct fully_layer *fully;
    size_t gsize, lsize;
    cl_kernel kern;
    cl_int err;

    g_assert (lay->type == LAYER_FULLY);

    fully = (struct fully_layer *) lay;
    lsize = 256;
    gsize = ceil ((float) lay->prev->size / lsize) * lsize;
    kern = fully->clear_input_gradient;

    clSetKernelArg (kern, 0, sizeof (cl_mem), &lay->prev->gradient_mem);
    err = clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                  kern, 1, NULL,
                                  &gsize, &lsize,
                                  0, NULL, NULL);
    g_assert (err == CL_SUCCESS);

    lsize = 256;
    gsize = ceil ((float) lay->size / lsize) * lsize;
    kern = fully->bias_backprop;

    clSetKernelArg (kern, 0, sizeof (cl_mem), &lay->gradient_mem);
    clSetKernelArg (kern, 1, sizeof (cl_mem), &lay->bias_delta_mem);
    clSetKernelArg (kern, 2, sizeof (cl_mem), &lay->bias_mem);
    clSetKernelArg (kern, 3, sizeof (cl_float), &lay->net->rate);
    clSetKernelArg (kern, 4, sizeof (cl_float), &lay->net->momentum);
    clSetKernelArg (kern, 5, sizeof (cl_float), &lay->net->decay);
    err = clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                  kern, 1, NULL,
                                  &gsize, &lsize,
                                  0, NULL, NULL);
    g_assert (err == CL_SUCCESS);

    lsize = lay->size;
    gsize = lay->size * lay->prev->size;
    kern = fully->weight_backprop;

    clSetKernelArg (kern, 0, sizeof (cl_mem), &lay->gradient_mem);
    clSetKernelArg (kern, 1, sizeof (cl_mem), &lay->delta_mem);
    clSetKernelArg (kern, 2, sizeof (cl_mem), &lay->weight_mem);
    clSetKernelArg (kern, 3, sizeof (cl_mem), &lay->value_mem);
    clSetKernelArg (kern, 4, sizeof (cl_mem), &lay->prev->gradient_mem);
    clSetKernelArg (kern, 5, sizeof (cl_mem), &lay->prev->value_mem);
    clSetKernelArg (kern, 6, sizeof (cl_float), &lay->net->rate);
    clSetKernelArg (kern, 7, sizeof (cl_float), &lay->net->momentum);
    clSetKernelArg (kern, 8, sizeof (cl_float), &lay->net->decay);
    err |= clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                   kern, 1, NULL,
                                   &gsize, &lsize,
                                   0, NULL, NULL);
    g_assert (err == CL_SUCCESS);

    lsize = 256;
    gsize = ceil((float) lay->prev->size / 256) * 256;
    kern = fully->derive_gradient;

    err = clSetKernelArg (kern, 0, sizeof (cl_mem), &lay->prev->value_mem);
    clSetKernelArg (kern, 1, sizeof (cl_mem), &lay->prev->gradient_mem);
    clEnqueueNDRangeKernel (lay->net->ctx->queue,
                            kern, 1, NULL,
                            &gsize, &lsize,
                            0, NULL, NULL);
    g_assert (err == CL_SUCCESS);
}

static void
release (struct layer *lay)
{
}
