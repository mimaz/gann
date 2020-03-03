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

static const char *forward_source = R"(
__kernel void reduce_inputs (__global const float *input,
                       __global const float *weight,
                       __global float *value) {
    __local float partial[INPUTS];

    int local_id = get_local_id (0);
    int global_id = get_global_id (0);
    int group_id = get_group_id (0);

    partial[local_id] = input[local_id] * weight[global_id];

    for (int off = get_local_size (0) / 2; off > 0; off /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id < off) {
            partial[local_id] += partial[local_id + off];
        }
    }

    if (local_id == 0) {
        value[group_id] = partial[0];
    }
}

__kernel void bias_activate (__global const float *bias,
                        __global float *value) {
    int global_id = get_global_id (0);

    value[global_id] = ACTIVATION (value[global_id] + bias[global_id]);
}
)";

static cl_program
build_program (struct network *net,
               const char *source,
               int inputs, int outputs)
{
    cl_program prog;
    cl_int err;
    char options[512], *log;
    size_t log_size;

    prog = clCreateProgramWithSource (net->ctx->context,
                                      1, &source,
                                      NULL, &err);
    g_assert (err == CL_SUCCESS);

    snprintf (options, sizeof (options),
              "-DINPUTS=%d -DOUTPUTS=%d -DACTIVATION(x)=\"x > 0 ? x : 0\" "
              "-DDERIVATIVE(a)=\"a > 0 ? 1 : 0\"",
              inputs, outputs);

    err = clBuildProgram (prog, 0, NULL, options, NULL, NULL);

    if (err != CL_SUCCESS) {
        clGetProgramBuildInfo (prog, net->ctx->device, 
                               CL_PROGRAM_BUILD_LOG,
                               0, NULL, &log_size);
        log = g_new (char, log_size);
        clGetProgramBuildInfo (prog, net->ctx->device,
                               CL_PROGRAM_BUILD_LOG,
                               log_size, log, NULL);
        g_message (log);
        g_free (log);
        g_assert (0);
    }

    return prog;
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
    base->value_mem = clCreateBuffer (net->ctx->context,
                                      CL_MEM_READ_WRITE,
                                      size * sizeof (cl_float),
                                      NULL, &err);
    g_assert (err == 0);
    base->gradient_v = g_new (float, size);
    base->gradient_mem = clCreateBuffer (net->ctx->context,
                                         CL_MEM_READ_WRITE,
                                         size * sizeof (cl_float),
                                         NULL, &err);
    g_assert (err == 0);
    base->bias_v = g_new (float, size);
    base->bias_mem = clCreateBuffer (net->ctx->context,
                                     CL_MEM_READ_WRITE,
                                     size * sizeof (cl_float),
                                     NULL, &err);
    g_assert (err == 0);
    base->weight_v = g_new (float, weights);
    base->weight_mem = clCreateBuffer (net->ctx->context,
                                       CL_MEM_READ_WRITE,
                                       weights * sizeof (cl_float),
                                       NULL, &err);
    g_assert (err == 0);
    base->delta_v = g_new (float, weights);
    base->delta_mem = clCreateBuffer (net->ctx->context,
                                      CL_MEM_READ_WRITE,
                                      weights * sizeof (cl_float),
                                      NULL, &err);
    g_assert (err == 0);
    base->bias_delta_v = g_new (float, weights);
    base->bias_delta_mem = clCreateBuffer (net->ctx->context,
                                           CL_MEM_READ_WRITE,
                                           weights * sizeof (cl_float),
                                           NULL, &err);
    g_assert (err == 0);
    base->program = build_program (net, forward_source,
                                   base->prev->size, base->size);
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
        base->weight_v[i] = (float) (i + 1) / weights / prev->size;

        if (activation == ACTIVATION_SIGMOID) {
            base->weight_v[i] -= 1.0f;
        }
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
    local_size = 256;
    global_size = ceil((float) lay->size / local_size) * local_size;
    err = clSetKernelArg (lay->kernels[KERNEL_BIAS_ACTIVATE], 0,
                          sizeof (cl_mem), &lay->prev->value_mem);
    err |= clSetKernelArg (lay->kernels[KERNEL_BIAS_ACTIVATE], 1,
                           sizeof (cl_mem), &lay->weight_mem);
    err |= clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                   lay->kernels[KERNEL_BIAS_ACTIVATE],
                                   1, NULL,
                                   &global_size, &local_size,
                                   0, NULL, NULL);
    g_assert (err == CL_SUCCESS);
    clFinish (lay->net->ctx->queue);

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
            + *gradient_p * lay->net->rate;
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
