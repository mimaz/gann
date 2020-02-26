#include "layer.h"
#include "network.h"

static void forward (struct layer *lay);

struct layer *
layer_make_input (struct network *net,
                  int width, int height, int depth)
{
    struct layer *base;

    base = g_new0 (struct layer, 1);

    base->net = net;
    base->type = LAYER_INPUT;
    base->weight_v = NULL;
    base->value_v = g_new (float, width * height * depth);
    base->weight_c = 0;
    base->value_c = width * height * depth;
    base->width = width;
    base->height = height;
    base->depth = depth;
    base->activation = ACTIVATION_NONE;
    base->forward = forward;
    base->backward = NULL;
    base->release = NULL;

    network_push_layer (net, base);

    return base;
}

static void
forward (struct layer *lay)
{
    g_assert (lay->net->input_c == lay->value_c);
    memcpy (lay->value_v, lay->net->input_v, lay->value_c * sizeof (float));
    network_set_data (lay->net, lay->value_v, lay->value_c);
}
