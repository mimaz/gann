#include "layer.h"
#include "network.h"

struct output_layer
{
    struct layer base;
    float *truth_v;
    float loss;
};

static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);

struct layer *
layer_make_output (struct network *net)
{
    struct output_layer *out;
    struct layer *base, *prev;
    int size;

    out = g_new0 (struct output_layer, 1);
    base = (struct layer *) out;
    prev = network_layer_last (net);
    size = prev->width * prev->height * prev->depth;

    base->net = net;
    base->prev = prev;
    base->type = LAYER_OUTPUT;
    base->activation = ACTIVATION_LINEAR;
    base->value_v = g_new (float, size);
    base->gradient_v = g_new (float, size);
    base->weight_v = NULL;
    base->delta_v = NULL;
    base->width = prev->width;
    base->height = prev->height;
    base->depth = prev->depth;
    base->size = size;
    base->weights = 0;
    base->forward = forward;
    base->backward = backward;
    base->release = release;

    network_push_layer (net, base);

    return base;
}

void
layer_output_set_truth (struct layer *lay,
                        const float *data,
                        int size)
{
    struct output_layer *out;

    g_assert (lay->type == LAYER_OUTPUT);
    g_assert (lay->size == size);

    out = (struct output_layer *) lay;

    g_clear_pointer (&out->truth_v, g_free);

    out->truth_v = g_memdup (data, size * sizeof (float));
}

static void
forward (struct layer *lay)
{
    g_assert (lay->size == lay->prev->size);

    memcpy (lay->value_v, lay->prev->value_v,
            sizeof (float) * lay->size);
}

static void
backward (struct layer *lay)
{
    struct output_layer *out;
    float sum, sub;
    int i;

    g_assert (lay->size == lay->prev->size);

    out = (struct output_layer *) lay;
    sum = 0;

    for (i = 0; i < lay->size; i++) {
        sub = out->truth_v[i] - lay->value_v[i];
        sum += sub * sub;
        /* g_message ("sub %f %f %f", sub, out->truth_v[i], lay->value_v[i]); */

        lay->gradient_v[i] = sub;
    }

    out->loss = sqrtf (sum);
    g_assert (out->loss == out->loss);

    lay->net->loss += out->loss;

    for (i = 0; i < lay->size; i++) {
        lay->prev->gradient_v[i] = lay->gradient_v[i] * out->loss;
        /* g_message ("prev gradient %f", lay->prev->gradient_v[i]); */
    }
}

static void
release (struct layer *lay)
{
    struct output_layer *out;

    out = (struct output_layer *) lay;

    g_clear_pointer (&lay->value_v, g_free);
    g_clear_pointer (&lay->gradient_v, g_free);
    g_clear_pointer (&out->truth_v, g_free);
}
