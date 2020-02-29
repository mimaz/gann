#include "layer.h"
#include "network.h"

static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);

struct layer *
layer_make_full (struct network *net,
                 enum activation_type activation,
                 int width, int height, int depth)
{
    struct layer *base, *prev;
    int size, weights, i;

    base = g_new0 (struct layer, 1);
    prev = network_layer_last (net);

    size = width * height * depth;
    weights = (prev->size + 1) * size;

    base->net = net;
    base->prev = prev;
    base->type = LAYER_FULLY;
    base->activation = activation;
    base->value_v = g_new (float, size);
    base->gradient_v = g_new (float, size);
    base->weight_v = g_new (float, weights);
    base->delta_v = g_new (float, weights);
    base->width = width;
    base->height = height;
    base->depth = depth;
    base->size = size;
    base->weights = weights;
    base->forward = forward;
    base->backward = backward;
    base->release = release;

    network_push_layer (net, base);

    for (i = 0; i < base->weights; i++) {
        base->weight_v[i] = (float) rand () / RAND_MAX - 0.5f;
        base->delta_v[i] = 0;
    }

    return base;
}

static void
forward (struct layer *lay)
{
    const float *input_p, *weight_p;
    float sum, *value_p;

    g_assert (lay->prev->size == lay->net->input_c);

    weight_p = lay->weight_v;
    value_p = lay->value_v;

    while (value_p < lay->value_v + lay->size) {
        input_p = lay->net->input_v;

        sum = *weight_p++;

        while (input_p < lay->net->input_v + lay->net->input_c) {
            sum += *weight_p++ * *input_p++;
        }

        *value_p++ = activation_value (lay->activation, sum);
    }

    g_assert (weight_p == lay->weight_v + lay->weights);

    network_set_data (lay->net, lay->value_v, lay->size);
}

static void
backward (struct layer *lay)
{
    struct network *net;
    float grad, der, *delta_p, *weight_p;
    int i, j;

    net = lay->net;
    weight_p = lay->weight_v;
    delta_p = lay->delta_v;

    for (j = 0; j < lay->prev->size; j++) {
        lay->prev->gradient_v[j] = 0;
    }

    for (i = 0; i < lay->size; i++) {
        /* bias */
        grad = lay->gradient_v[i];

        *delta_p = *delta_p * net->momentum + grad * net->rate;
        *weight_p += *delta_p;

        delta_p++;
        weight_p++;

        for (j = 0; j < lay->prev->size; j++) {
            *delta_p = *delta_p
                * net->momentum
                + grad * net->rate * lay->prev->value_v[j];
            *weight_p = *weight_p * net->decay + *delta_p;

            lay->prev->gradient_v[j] += grad * *weight_p;

            weight_p++;
            delta_p++;
        }
    }

    for (j = 0; j < lay->prev->size; j++) {
        der = activation_derivative (lay->prev->activation,
                                     lay->prev->value_v[j]);
        lay->prev->gradient_v[j] *= der;
    }
}

static void
release (struct layer *lay)
{
    g_clear_pointer (&lay->value_v, g_free);
    g_clear_pointer (&lay->gradient_v, g_free);
    g_clear_pointer (&lay->weight_v, g_free);
    g_clear_pointer (&lay->delta_v, g_free);
}
