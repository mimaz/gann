#include "layer.h"
#include "network.h"

struct output_layer
{
    struct layer base;
    cl_mem truth_mem;
    cl_mem loss_mem;
    cl_event truth_event;
    cl_kernel backprop_kern;
    float loss;
};

static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);

struct layer *
layer_make_output (struct network *net)
{
    struct output_layer *out;
    struct layer *base, *prev;
    int size;
    char options[128];
    cl_program program;

    out = g_new0 (struct output_layer, 1);
    base = (struct layer *) out;
    prev = network_layer_last (net);
    size = prev->width * prev->height * prev->depth;

    g_snprintf (options, sizeof (options),
                "-DINPUTS=%d -DOUTPUTS=%d",
                prev->size, size);
    program = context_build_program (net->ctx,
                                     options,
                                     "output-layer.cl",
                                     NULL);

    base->net = net;
    base->prev = prev;
    base->type = LAYER_OUTPUT;
    base->activation = ACTIVATION_LINEAR;
    base->program = program;
    base->width = prev->width;
    base->height = prev->height;
    base->depth = prev->depth;
    base->size = size;
    base->weights = 0;
    base->forward = forward;
    base->backward = backward;
    base->release = release;

    layer_create_buffer (base, &base->value_mem, size,
                         CL_MEM_READ_WRITE);
    layer_create_buffer (base, &out->truth_mem, size,
                         CL_MEM_READ_ONLY);
    layer_create_buffer (base, &out->loss_mem, 1,
                         CL_MEM_WRITE_ONLY);
    layer_create_kernel (base, &out->backprop_kern, "backprop");

    network_push_layer (net, base);

    return base;
}

void
layer_output_set_truth (struct layer *lay,
                        const float *data,
                        int size)
{
    struct output_layer *out;

    g_assert (lay->type == LAYER_OUTPUT);
    g_assert (lay->size == size);

    out = (struct output_layer *) lay;

    clEnqueueWriteBuffer (lay->net->ctx->queue,
                          out->truth_mem,
                          CL_FALSE,
                          0, size * sizeof (cl_float),
                          data, 0, NULL,
                          &out->truth_event);
}

static void
forward (struct layer *lay)
{
    g_assert (lay->type == LAYER_OUTPUT);
    g_assert (lay->size == lay->prev->size);

    lay->value_mem = lay->prev->value_mem;
}

static void
backward (struct layer *lay)
{
    struct output_layer *out;
    size_t globsiz, locsiz;
    cl_kernel kern;
    cl_int err;

    g_assert (lay->type == LAYER_OUTPUT);
    g_assert (lay->size == lay->prev->size);

    out = (struct output_layer *) lay;
    kern = out->backprop_kern;

    clSetKernelArg (kern, 0, sizeof (cl_mem), &out->truth_mem);
    clSetKernelArg (kern, 1, sizeof (cl_mem), &lay->value_mem);
    clSetKernelArg (kern, 2, sizeof (cl_mem), &lay->prev->gradient_mem);
    clSetKernelArg (kern, 3, sizeof (cl_mem), &out->loss_mem);

    locsiz = lay->size;
    globsiz = locsiz;
    err = clEnqueueNDRangeKernel (lay->net->ctx->queue,
                                  kern, 1, NULL,
                                  &globsiz, &locsiz,
                                  1, &out->truth_event,
                                  NULL);
    g_assert (err == CL_SUCCESS);

    clEnqueueReadBuffer (lay->net->ctx->queue,
                         out->loss_mem,
                         CL_TRUE,
                         0, sizeof (cl_float),
                         &out->loss, 0, NULL, NULL);
    lay->net->loss = out->loss;
}

static void
release (struct layer *lay)
{
}
