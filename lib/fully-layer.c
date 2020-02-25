#include "layer.h"
#include "network.h"

static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);

struct layer *
layer_make_full (struct network *net,
                 int width, int height, int depth)
{
    struct layer *base, *prev;
    int insize, outsize;

    base = g_new0 (struct layer, 1);
    prev = network_layer_last (net);

    insize = prev->width * prev->height * prev->depth;
    outsize = width * height * depth;

    base->net = net;
    base->type = LAYER_FULLY;
    base->weight_v = g_new (float, insize * outsize);
    base->value_v = g_new (float, outsize);
    base->bias_v = g_new (float, outsize);
    base->weight_c = insize * outsize;
    base->value_c = outsize;
    base->bias_c = outsize;
    base->width = width;
    base->height = height;
    base->depth = depth;
    base->forward = forward;
    base->backward = backward;
    base->release = release;

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
}
