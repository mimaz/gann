#include "layer.h"
#include "network.h"

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
forward_linear (const float *in, float *out, int size)
{
}

static void
forward_relu (const float *in, float *out, int size)
{
}

static void
forward (struct layer *lay)
{
    static const void (*forward_v[]) (const float *, float *, int) = {
        [ACTIVATION_NONE] = forward_linear,
        [ACTIVATION_LINEAR] = forward_linear,
        [ACTIVATION_RELU] = forward_relu,
    };

    struct activation_layer *acti;
    void (*forward_f) (const float *, float *, int);

    g_assert (lay->net->input_c == lay->value_c);
    g_assert (lay->type == LAYER_ACTIVATION);

    acti = (struct activation_layer *) lay;
    g_assert (acti->activation < N_ACTIVATIONS);

    forward_f = forward_v[acti->activation];
    g_assert (forward_f != NULL);
    forward_f (lay->net->input_v, lay->value_v, lay->value_c);

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
