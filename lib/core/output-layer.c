#include "layer.h"
#include "network.h"

enum
{
    K_CALC_ERROR,
    N_KERNELS,
};

static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);

struct layer *
layer_make_output (struct network *net)
{
    struct layer *base, *prev;
    int size;
    g_autofree char *options;
    cl_program program;
    cl_int err;

    base = g_new0 (struct layer, 1);
    prev = network_layer_last (net);
    size = prev->width * prev->height * prev->depth;

    options = g_strdup_printf ("-DINPUTS=%d -DOUTPUTS=%d",
                               prev->size, size);
    program = context_build_program (net->ctx,
                                     options,
                                     "output-layer.cl",
                                     NULL);

    base->net = net;
    base->prev = prev;
    base->type = LAYER_OUTPUT;
    base->activation = ACTIVATION_LINEAR;
    base->value_mem = clCreateBuffer (net->ctx->context,
                                      CL_MEM_READ_WRITE,
                                      size * sizeof (cl_float),
                                      NULL, &err);
    g_assert (err == CL_SUCCESS);
    base->gradient_mem = clCreateBuffer (net->ctx->context,
                                         CL_MEM_READ_WRITE,
                                         size * sizeof (cl_float),
                                         NULL, &err);
    g_assert (err == CL_SUCCESS);
    base->truth_mem = clCreateBuffer (net->ctx->context,
                                      CL_MEM_READ_ONLY,
                                      size * sizeof (cl_float),
                                      NULL, &err);
    g_assert (err == CL_SUCCESS);
    base->weight_mem = 0;
    base->delta_mem = 0;
    base->bias_delta_mem = 0;
    base->program = program;
    base->width = prev->width;
    base->height = prev->height;
    base->depth = prev->depth;
    base->size = size;
    base->weights = 0;
    base->forward = forward;
    base->backward = backward;
    base->release = release;

    base->loss_mem = clCreateBuffer (net->ctx->context,
                                    CL_MEM_READ_WRITE,
                                    sizeof (cl_float),
                                    NULL, &err);

    layer_create_kernel (base, K_CALC_ERROR, "calc_error");

    network_push_layer (net, base);

    return base;
}

void
layer_output_set_truth (struct layer *lay,
                        const float *data,
                        int size)
{
    g_assert (lay->type == LAYER_OUTPUT);
    g_assert (lay->size == size);

    clEnqueueWriteBuffer (lay->net->ctx->queue,
                          lay->truth_mem,
                          CL_FALSE,
                          0, size * sizeof (cl_float),
                          data,
                          0, NULL, NULL);
}

static void
forward (struct layer *lay)
{
    g_assert (lay->size == lay->prev->size);
    lay->value_mem = lay->prev->value_mem;
}

static void
backward (struct layer *lay)
{
    g_assert (lay->size == lay->prev->size);

    cl_int err;
    size_t globsize, locsize;

    g_assert (lay->size == lay->prev->size);

    err = clSetKernelArg (lay->kernels[K_CALC_ERROR], 0,
                          sizeof (cl_mem), &lay->truth_mem);
    err |= clSetKernelArg (lay->kernels[K_CALC_ERROR], 1,
                           sizeof (cl_mem), &lay->value_mem);
    err |= clSetKernelArg (lay->kernels[K_CALC_ERROR], 2,
                           sizeof (cl_mem), &lay->gradient_mem);
    err |= clSetKernelArg (lay->kernels[K_CALC_ERROR], 3,
                           sizeof (cl_mem), &lay->prev->gradient_mem);
    err |= clSetKernelArg (lay->kernels[K_CALC_ERROR], 4,
                           sizeof (cl_mem), &lay->loss_mem);

    locsize = lay->size;
    globsize = locsize;
    err |= clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                   lay->kernels[K_CALC_ERROR],
                                    1, NULL,
                                    &globsize, &locsize,
                                    0, NULL, NULL);

    float loss;
    clEnqueueReadBuffer (lay->net->ctx->queue,
                         lay->loss_mem,
                         CL_FALSE,
                         0, sizeof (cl_float),
                         &loss, 0, NULL, NULL);
    lay->net->loss = loss;
}

static void
release (struct layer *lay)
{
}
