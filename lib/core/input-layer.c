#include "layer.h"
#include "network.h"
#include "context.h"

struct layer *
layer_make_input (struct network *net,
                  int width, int height, int depth)
{
    struct layer *base;
    int size;

    base = g_new0 (struct layer, 1);
    size = width * height * depth;

    base->net = net;
    base->type = LAYER_INPUT;
    base->activation = ACTIVATION_LINEAR;
    base->width = width;
    base->height = height;
    base->depth = depth;
    base->size = size;
    base->weights = 0;

    layer_create_buffer (base, &base->value_mem,
                         size, CL_MEM_READ_WRITE);
    layer_create_buffer (base, &base->gradient_mem,
                         size, CL_MEM_READ_WRITE);

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

    g_autofree float *buff = g_memdup (data, sizeof (float) * size);

    for (int i = 0; i < size; i++)
        buff[i] = data[i] - 0.5f;

    clEnqueueWriteBuffer (lay->net->ctx->queue,
                          lay->value_mem,
                          CL_TRUE,
                          0, size * sizeof (cl_float),
                          buff,
                          0, NULL, NULL);
    clFinish (lay->net->ctx->queue);
}
