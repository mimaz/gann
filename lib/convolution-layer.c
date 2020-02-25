#include "layer.h"
#include "network.h"

#include <glib-object.h>

struct convolution_layer
{
    struct layer base;
    int kernel_area;
};

static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);

struct layer *
layer_make_convolution (struct network *net,
                        int range, int filterc)
{
    struct convolution_layer *conv;
    struct layer *base, *prev;
    int width, height, depth, karea, weightc, size;

    conv = g_new0 (struct convolution_layer, 1);
    base = (struct layer *) conv;

    prev = network_layer_last (net);

    width = prev->width;
    height = prev->height;
    depth = prev->depth;

    karea = (range * 2 + 1) * (range * 2 + 1);
    size = width * height * filterc;
    weightc = size * karea * depth;

    base->net = net;
    base->type = LAYER_CONVOLUTION;
    base->weight_v = g_new (float, weightc);
    base->value_v = g_new (float, size);
    base->bias_v = NULL;
    base->weight_c = weightc;
    base->value_c = size;
    base->bias_c = 0;
    base->width = width;
    base->height = height;
    base->depth = filterc;
    base->forward = forward;
    base->backward = backward;
    base->release = release;

    conv->kernel_area = karea;

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
    g_clear_pointer (&lay->weight_v, g_free);
    g_clear_pointer (&lay->value_v, g_free);
    g_clear_pointer (&lay->bias_v, g_free);
}
