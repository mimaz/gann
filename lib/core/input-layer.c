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

    base = g_new0 (struct layer, 1);
    size = width * height * depth;

    base->net = net;
    base->type = LAYER_INPUT;
    base->activation = ACTIVATION_LINEAR;
    base->value_v = g_new (float, size);
    base->gradient_v = g_new (float, size);
    base->weight_v = NULL;
    base->delta_v = NULL;
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

static void
forward (struct layer *lay)
{
    g_assert (lay->net->input_c == lay->size);
    memcpy (lay->value_v, lay->net->input_v, lay->size * sizeof (float));
    network_set_data (lay->net, lay->value_v, lay->size);
}

static void
backward (struct layer *lay)
{
}

static void
release (struct layer *lay)
{
    g_clear_pointer (&lay->value_v, g_free);
    g_clear_pointer (&lay->gradient_v, g_free);
}
