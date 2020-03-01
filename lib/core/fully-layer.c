#include "layer.h"
#include "network.h"
#include "context.h"

#define USE_OPENCL

static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);

static const char *forward_source = R"(
__kernel void forward (__global float *input_v,
                       __global float *value_v,
                       __global float *weight_v,
                       unsigned int size,
                       unsigned int weights) {
    for (int i = 0; i < size; i++) {
        input_v[i] = i;
    }
}

__kernel void backward () {
}
)";

static cl_program
build_program (struct network *net,
               int sourcec,
               const char **sourcev)
{
    cl_program prog;
    cl_int err;
    char *log;
    size_t log_size;

    prog = clCreateProgramWithSource (net->ctx->context,
                                      sourcec, sourcev,
                                      NULL, &err);
    g_assert (err == CL_SUCCESS);

    err = clBuildProgram (prog, 0, NULL, NULL, NULL, NULL);

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
    weights = (prev->size + 1) * size;

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
    base->program = build_program (net, 1, &forward_source);
    base->forward_kernel = clCreateKernel (base->program, "forward", &err);
    g_assert (err == CL_SUCCESS);
    base->backward_kernel = clCreateKernel (base->program, "backward", &err);
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

    return base;
}

static void
forward (struct layer *lay)
{
#ifdef USE_OPENCL
    cl_mem input_mem;
    cl_int err;
    size_t global_size, local_size;

    global_size = lay->size;
    local_size = 1;
    input_mem = clCreateBuffer (lay->net->ctx->context,
                                CL_MEM_READ_WRITE,
                                lay->size * sizeof (cl_float),
                                NULL, &err);
    g_assert (err == CL_SUCCESS);

    clSetKernelArg (lay->forward_kernel, 0,
                    sizeof (cl_mem), &input_mem);
    clSetKernelArg (lay->forward_kernel, 1,
                    sizeof (cl_mem), &lay->value_mem);
    clSetKernelArg (lay->forward_kernel, 2,
                    sizeof (cl_mem), &lay->weight_mem);
    clSetKernelArg (lay->forward_kernel, 3,
                    sizeof (cl_uint), &lay->size);
    clSetKernelArg (lay->forward_kernel, 4,
                    sizeof (cl_uint), &lay->weights);
    clEnqueueNDRangeKernel (lay->net->ctx->queue,
                            lay->forward_kernel,
                            1, NULL, &global_size, &local_size,
                            0, NULL, NULL);
    clFinish (lay->net->ctx->queue);

    clReleaseMemObject (input_mem);

    g_message ("done");
    exit (0);
#else
    const float *input_p, *weight_p;
    float sum, *value_p;

    weight_p = lay->weight_v;
    value_p = lay->value_v;

    while (value_p < lay->value_v + lay->size) {
        input_p = lay->prev->value_v;

        sum = *weight_p++;

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
    float *delta_p, *weight_p, *gradient_p;
    int i, j;

    weight_p = lay->weight_v;
    delta_p = lay->delta_v;
    gradient_p = lay->gradient_v;

    for (j = 0; j < lay->prev->size; j++) {
        lay->prev->gradient_v[j] = 0;
    }

    for (i = 0; i < lay->size; i++) {
        /* bias */
        *delta_p = *delta_p * lay->net->momentum
            + *gradient_p * lay->net->rate;
        *weight_p = *weight_p * lay->net->decay + *delta_p;

        delta_p++;
        weight_p++;

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
