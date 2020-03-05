#include "layer.h"
#include "network.h"
#include "context.h"

static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);

struct layer *
layer_make_input (struct network *net,
                  int width, int height, int depth)
{
    struct layer *base;
    int size;
    cl_int err;

    base = g_new0 (struct layer, 1);
    size = width * height * depth;

    base->net = net;
    base->type = LAYER_INPUT;
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
    base->width = width;
    base->height = height;
    base->depth = depth;
    base->size = size;
    base->weights = 0;
    base->forward = forward;
    base->backward = backward;
    base->release = release;

    network_push_layer (net, base);

    return base;
}

void
layer_input_set_data (struct layer *lay,
                      const float *data,
                      int size)
{
    g_assert (lay->type == LAYER_INPUT);
    g_assert (size == lay->size);

    clEnqueueWriteBuffer (lay->net->ctx->queue,
                          lay->value_mem,
                          CL_FALSE,
                          0, size * sizeof (cl_float),
                          data,
                          0, NULL, NULL);
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
