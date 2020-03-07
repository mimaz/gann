#include "layer.h"
#include "network.h"
#include "context.h"

#include <stdio.h>

#define USE_OPENCL

struct fully_layer
{
    struct layer base;
    cl_kernel forward;
    cl_kernel derive_gradient;
    cl_kernel backward;
    cl_kernel backward_bias;
};

static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);

struct layer *
layer_make_full (struct network *net,
                 enum activation_type activation,
                 int width, int height, int depth)
{
    struct fully_layer *fully;
    struct layer *base, *prev;
    g_autoptr (GString) options;
    cl_program prog;
    int size, weights, i;

    g_assert (sizeof (cl_float) == sizeof (gfloat));

    fully = g_new0 (struct fully_layer, 1);
    base = (struct layer *) fully;
    prev = network_layer_last (net);

    size = width * height * depth;
    weights = prev->size * size;

    options = g_string_new (NULL);

    g_string_append_printf (options, "-DINPUTS=%d ", prev->size);
    g_string_append_printf (options, "-DOUTPUTS=%d ", size);

    if (prev->gradient_mem != 0) {
        g_string_append (options, "-DCALC_GRADIENT ");
    }

    prog = context_build_program (net->ctx,
                                  options->str,
                                  "fully-layer.cl",
                                  NULL);

    base->net = net;
    base->prev = prev;
    base->type = LAYER_FULLY;
    base->activation = activation;
    base->program = prog;
    base->width = width;
    base->height = height;
    base->depth = depth;
    base->size = size;
    base->weights = weights;
    base->forward = forward;
    base->backward = backward;
    base->release = release;

    layer_create_buffer (base, &base->value_mem,
                         size, CL_MEM_READ_WRITE);
    layer_create_buffer (base, &base->gradient_mem,
                         size, CL_MEM_READ_WRITE);
    layer_create_buffer (base, &base->bias_mem,
                         size, CL_MEM_READ_WRITE);
    layer_create_buffer (base, &base->bias_delta_mem,
                         size, CL_MEM_READ_WRITE);
    layer_create_buffer (base, &base->weight_mem,
                         weights, CL_MEM_READ_WRITE);
    layer_create_buffer (base, &base->delta_mem,
                         weights, CL_MEM_READ_WRITE);

    layer_create_kernel (base, &fully->forward, "forward");
    layer_create_kernel (base, &fully->derive_gradient, "derive_gradient");
    layer_create_kernel (base, &fully->backward, "backward");
    layer_create_kernel (base, &fully->backward_bias, "backward_bias");

    network_push_layer (net, base);

    g_autofree float *gradient_v = g_new (float, size);
    g_autofree float *bias_v = g_new (float, size);
    g_autofree float *bias_delta_v = g_new (float, size);
    g_autofree float *delta_v = g_new (float, weights);
    g_autofree float *weight_v = g_new (float, weights);

    for (i = 0; i < weights; i++) {
        float r1 = cosf (2.0f * (float) M_PI * rand () / RAND_MAX);
        float r2 = sqrtf (-2.0f  * logf ((float) rand () / RAND_MAX));
        float d = (r1 * r2) * sqrtf (2.0f / prev->size);

        weight_v[i] = d;
        delta_v[i] = 0;
    }

    for (i = 0; i < size; i++) {
        bias_v[i] = 0;
        bias_delta_v[i] = 0;
        gradient_v[i] = 0;
    }

    clEnqueueWriteBuffer (net->ctx->queue,
                          base->gradient_mem,
                          CL_TRUE,
                          0, size * sizeof (cl_float),
                          gradient_v,
                          0, NULL, NULL);
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
    size_t globsiz, locsiz;
    cl_kernel kern;
    cl_int err;

    g_assert (lay->type == LAYER_FULLY);

    fully = (struct fully_layer *) lay;

    locsiz = MIN (lay->size, lay->net->ctx->group_size);
    globsiz = ceilf ((float) lay->size / locsiz) * locsiz;
    kern = fully->forward;

    clSetKernelArg (kern, 0, sizeof (cl_mem), &lay->prev->value_mem);
    clSetKernelArg (kern, 1, sizeof (cl_mem), &lay->weight_mem);
    clSetKernelArg (kern, 2, sizeof (cl_mem), &lay->bias_mem);
    clSetKernelArg (kern, 3, sizeof (cl_mem), &lay->value_mem);

    clFinish (lay->net->ctx->queue);
    err = clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                  kern, 1, NULL,
                                  &globsiz, &locsiz,
                                  0, NULL, NULL);
    g_assert (err == CL_SUCCESS);
    clFinish (lay->net->ctx->queue);
}

static void
backward (struct layer *lay)
{
    struct fully_layer *fully;
    size_t globsiz, locsiz;
    float ratefactor;
    cl_kernel kern;
    cl_int err;

    g_assert (lay->type == LAYER_FULLY);
    ratefactor = lay->net->rate * (1 - lay->net->momentum);
    fully = (struct fully_layer *) lay;

    locsiz = MIN (lay->size, lay->net->ctx->group_size);
    globsiz = ceilf ((float) lay->prev->size / locsiz) * locsiz;
    kern = fully->derive_gradient;

    clSetKernelArg (kern, 0, sizeof (cl_mem), &lay->value_mem);
    clSetKernelArg (kern, 1, sizeof (cl_mem), &lay->gradient_mem);

    err = clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                  kern, 1, NULL,
                                  &globsiz, &locsiz,
                                  0, NULL, NULL);
    g_assert (err == CL_SUCCESS);

    locsiz = lay->net->ctx->group_size;
    globsiz = ceilf ((float) lay->prev->size / locsiz) * locsiz;
    kern = fully->backward;

    clSetKernelArg (kern, 0, sizeof (cl_mem), &lay->prev->value_mem);
    clSetKernelArg (kern, 1, sizeof (cl_mem), &lay->gradient_mem);
    clSetKernelArg (kern, 2, sizeof (cl_mem), &lay->prev->gradient_mem);
    clSetKernelArg (kern, 3, sizeof (cl_mem), &lay->weight_mem);
    clSetKernelArg (kern, 4, sizeof (cl_mem), &lay->delta_mem);
    clSetKernelArg (kern, 5, sizeof (cl_float), &ratefactor);
    clSetKernelArg (kern, 6, sizeof (cl_float), &lay->net->momentum);
    clSetKernelArg (kern, 7, sizeof (cl_float), &lay->net->decay);

    clFinish (lay->net->ctx->queue);
    err = clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                  kern, 1, NULL,
                                  &globsiz, &locsiz,
                                  0, NULL, NULL);
    clFinish (lay->net->ctx->queue);
    g_assert (err == CL_SUCCESS);

    g_autofree float *buff = g_new (float, lay->size);
    clFinish (lay->net->ctx->queue);
    g_assert (lay->gradient_mem != 0);

    globsiz = ceilf ((float) lay->size / locsiz) * locsiz;
    kern = fully->backward_bias;

    clSetKernelArg (kern, 0, sizeof (cl_mem), &lay->gradient_mem);
    clSetKernelArg (kern, 1, sizeof (cl_mem), &lay->bias_mem);
    clSetKernelArg (kern, 2, sizeof (cl_mem), &lay->bias_delta_mem);
    clSetKernelArg (kern, 3, sizeof (cl_float), &ratefactor);
    clSetKernelArg (kern, 4, sizeof (cl_float), &lay->net->momentum);
    clSetKernelArg (kern, 5, sizeof (cl_float), &lay->net->decay);

    err = clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                  kern, 1, NULL,
                                  &globsiz, &locsiz,
                                  0, NULL, NULL);
    g_assert (err == CL_SUCCESS);
    clFinish (lay->net->ctx->queue);
}

static void
release (struct layer *lay)
{
}
