#include "layer.h"
#include "network.h"

#include <math.h>

struct activation_layer
{
    struct layer base;
    enum activation_type activation;
};

static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);

struct layer *
layer_make_activation (struct network *net,
                       enum activation_type activation)
{
    struct activation_layer *acti;
    struct layer *base, *prev;
    int size;

    acti = g_new0 (struct activation_layer, 1);
    base = (struct layer *) acti;
    prev = network_layer_last (net);

    size = prev->width * prev->height * prev->depth;

    base->net = net;
    base->type = LAYER_ACTIVATION;
    base->weight_v = NULL;
    base->value_v = g_new (float, size);
    base->bias_v = NULL;
    base->weight_c = 0;
    base->value_c = size;
    base->bias_c = 0;
    base->width = prev->width;
    base->height = prev->height;
    base->depth = prev->depth;
    base->forward = forward;
    base->backward = backward;
    base->release = release;

    acti->activation = activation;

    network_push_layer (net, base);

    return base;
}

static void
forward (struct layer *lay)
{
    struct activation_layer *act;
    float in, out;
    int i;

    act = (struct activation_layer *) lay;

    g_assert (lay->net->input_c == lay->value_c);

    for (i = 0; i < lay->value_c; i++) {
        in = lay->net->input_v[i];

        switch (act->activation) {
        case ACTIVATION_NONE:
        case ACTIVATION_LINEAR:
            out = in;
            break;

        case ACTIVATION_RELU:
            out = MAX (0, in);
            break;

        case ACTIVATION_SIGMOID:
            out = 1.0f / (1.0f + expf (-in));
            break;

        default:
            g_abort ();
            break;
        }

        lay->value_v[i] = out;
    }

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
