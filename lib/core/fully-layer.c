#include "layer.h"
#include "network.h"
#include "context.h"

#include <stdio.h>

#define USE_OPENCL

enum
{
    KERNEL_REDUCE_INPUTS,
    KERNEL_BIAS_ACTIVATE,
};

static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);

static cl_program
build_program (struct network *net,
               int inputs, int outputs)
{
    cl_program prog;
    cl_int err;
    char *opts, *log;
    size_t log_size;
    const char *src;

    src = context_read_cl_code (net->ctx, "fully-layer.cl");
    prog = clCreateProgramWithSource (net->ctx->context,
                                      1, &src,
                                      NULL, &err);
    g_assert (err == CL_SUCCESS);

    opts = g_strdup_printf ("-DINPUTS=%d "
                            "-DOUTPUTS=%d ",
                            inputs,
                            outputs);

    err = clBuildProgram (prog, 0, NULL, opts, NULL, NULL);
    g_free (opts);

    if (err != CL_SUCCESS) {
        clGetProgramBuildInfo (prog, net->ctx->device, 
                               CL_PROGRAM_BUILD_LOG,
                               0, NULL, &log_size);
        log = g_new (char, log_size);
        clGetProgramBuildInfo (prog, net->ctx->device,
                               CL_PROGRAM_BUILD_LOG,
                               log_size, log, NULL);
        g_error (log);
        g_free (log);
    }

    return prog;
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
    cl_int err;


    g_assert (sizeof (cl_float) == sizeof (gfloat));

    base = g_new0 (struct layer, 1);
    prev = network_layer_last (net);

    size = width * height * depth;
    weights = prev->size * size;

    base->net = net;
    base->prev = prev;
    base->type = LAYER_FULLY;
    base->activation = activation;
    base->value_v = g_new (float, size);
    base->value_mem = build_buffer (net, CL_MEM_READ_WRITE, size);
    base->gradient_v = g_new (float, size);
    base->gradient_mem = build_buffer (net, CL_MEM_READ_WRITE, size);
    base->bias_v = g_new (float, size);
    base->bias_mem = build_buffer (net, CL_MEM_READ_WRITE, size);
    base->weight_v = g_new (float, weights);
    base->weight_mem = build_buffer (net, CL_MEM_READ_WRITE, weights);
    base->delta_v = g_new (float, weights);
    base->delta_mem = build_buffer (net, CL_MEM_READ_WRITE, weights);
    base->bias_delta_v = g_new (float, weights);
    base->bias_delta_mem = build_buffer (net, CL_MEM_READ_WRITE, weights);
    base->program = build_program (net, base->prev->size, base->size);
    base->kernels[KERNEL_REDUCE_INPUTS] =
        clCreateKernel (base->program, "reduce_inputs", &err);
    base->kernels[KERNEL_BIAS_ACTIVATE] =
        clCreateKernel (base->program, "bias_activate", &err);
    g_assert (err == CL_SUCCESS);
    base->width = width;
    base->height = height;
    base->depth = depth;
    base->size = size;
    base->weights = weights;
    base->forward = forward;
    base->backward = backward;
    base->release = release;

    network_push_layer (net, base);

    for (i = 0; i < weights; i++) {
        base->delta_v[i] = 0;
        base->weight_v[i] = (float) rand () / RAND_MAX / prev->size;
    }

    for (i = 0; i < size; i++) {
        base->bias_v[i] = 0;
        base->bias_delta_v[i] = 0;
    }

    return base;
}

static void
forward (struct layer *lay)
{
#ifdef USE_OPENCL
    cl_int err = CL_SUCCESS;
    size_t global_size, local_size;

    local_size = lay->prev->size;
    global_size = lay->size * local_size;
    clEnqueueWriteBuffer (lay->net->ctx->queue,
                          lay->bias_mem,
                          CL_TRUE,
                          0, lay->size * sizeof (cl_float),
                          lay->bias_v,
                          0, NULL, NULL);
    clEnqueueWriteBuffer (lay->net->ctx->queue,
                          lay->weight_mem,
                          CL_TRUE,
                          0, lay->weights * sizeof (cl_float),
                          lay->weight_v,
                          0, NULL, NULL);
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
    local_size = 16;
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

    clEnqueueReadBuffer (lay->net->ctx->queue,
                         lay->value_mem,
                         CL_TRUE,
                         0, lay->size * sizeof (cl_float),
                         lay->value_v, 0, NULL, NULL);

#else
    const float *input_p, *weight_p, *bias_p;
    float sum, *value_p;

    weight_p = lay->weight_v;
    value_p = lay->value_v;
    bias_p = lay->bias_v;

    while (value_p < lay->value_v + lay->size) {
        input_p = lay->prev->value_v;

        sum = *bias_p++;

        while (input_p < lay->prev->value_v + lay->prev->size) {
            sum += *weight_p++ * *input_p++;
        }

        *value_p++ = activation_value (lay->activation, sum);
    }

    g_assert (weight_p == lay->weight_v + lay->weights);
#endif
}

static void
backward (struct layer *lay)
{
    float *bias_delta_p, *delta_p, *bias_p, *weight_p, *gradient_p;
    int i, j;

    weight_p = lay->weight_v;
    bias_p = lay->bias_v;
    bias_delta_p = lay->bias_delta_v;
    delta_p = lay->delta_v;
    gradient_p = lay->gradient_v;

    for (j = 0; j < lay->prev->size; j++) {
        lay->prev->gradient_v[j] = 0;
    }

    for (i = 0; i < lay->size; i++) {
        *bias_delta_p = *bias_delta_p * lay->net->momentum
            + *gradient_p * lay->net->rate * (1 - lay->net->momentum);
        *bias_p = *bias_p * lay->net->decay + *bias_delta_p;

        bias_delta_p++;
        bias_p++;

        for (j = 0; j < lay->prev->size; j++) {
            *delta_p = *delta_p
                * lay->net->momentum
                + *gradient_p * lay->net->rate * lay->prev->value_v[j];
            *weight_p = *weight_p * lay->net->decay + *delta_p;

            lay->prev->gradient_v[j] += *gradient_p * *weight_p;

            weight_p++;
            delta_p++;
        }

        gradient_p++;
    }

    for (j = 0; j < lay->prev->size; j++) {
        lay->prev->gradient_v[j] *=
            activation_derivative (lay->prev->activation,
                                   lay->prev->value_v[j]);
    }
}

static void
release (struct layer *lay)
{
    g_clear_pointer (&lay->value_mem, clReleaseMemObject);
    g_clear_pointer (&lay->gradient_mem, clReleaseMemObject);
    g_clear_pointer (&lay->weight_mem, clReleaseMemObject);
    g_clear_pointer (&lay->delta_mem, clReleaseMemObject);
    g_clear_pointer (&lay->value_v, g_free);
    g_clear_pointer (&lay->gradient_v, g_free);
    g_clear_pointer (&lay->weight_v, g_free);
    g_clear_pointer (&lay->delta_v, g_free);
}
