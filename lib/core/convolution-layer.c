#include "layer.h"
#include "network.h"

#include <glib-object.h>

struct convolution_layer
{
    struct layer base;
    int kernel_width;
    int kernel_height;
};

static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);

struct layer *
layer_make_convolution (struct network *net,
                        enum activation_type activation,
                        int kernel_width, int kernel_height,
                        int filter_count)
{
    struct convolution_layer *conv;
    struct layer *base, *prev;
    int width, height, depth, kernel_area, weight_c, value_c;

    conv = g_new0 (struct convolution_layer, 1);
    base = (struct layer *) conv;
    prev = network_layer_last (net);

    width = prev->width;
    height = prev->height;
    depth = prev->depth;

    kernel_area = kernel_width * kernel_height;
    value_c = width * height * filter_count;
    weight_c = value_c * kernel_area * depth;

    base->net = net;
    base->type = LAYER_CONVOLUTION;
    base->weight_v = g_new (float, weight_c);
    base->value_v = g_new (float, value_c);
    base->weight_c = weight_c;
    base->value_c = value_c;
    base->width = width;
    base->height = height;
    base->depth = filter_count;
    base->activation = activation;
    base->forward = forward;
    base->backward = backward;
    base->release = release;

    conv->kernel_width = kernel_width;
    conv->kernel_height = kernel_height;

    network_push_layer (net, base);

    return base;
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
