#include "layer.h"
#include "network.h"

static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);

struct layer *
layer_make_input (struct network *net,
                  int width, int height, int depth)
{
    struct layer *base;
    int size;

    size = width * width * depth;

    base = g_new0 (struct layer, 1);
    base->net = net;
    base->type = LAYER_INPUT;
    base->weight_v = NULL;
    base->value_v = g_new (float, size);
    base->bias_v = NULL;
    base->weight_c = 0;
    base->value_c = size;
    base->bias_c = 0;
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
    g_assert (lay->net->input_c == lay->value_c);
    memcpy (lay->value_v, lay->net->input_v, sizeof (float) * lay->value_c);
    network_set_data (lay->net, lay->value_v, lay->value_c);
}

static void
backward (struct layer *lay)
{
}

static void
release (struct layer *lay)
{
    g_clear_pointer (&lay->value_v, g_free);
}
