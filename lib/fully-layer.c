#include "layer.h"
#include "network.h"

static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);
static void loss (struct layer *lay);

struct fully_layer
{
    struct layer base;
    int insize;
    int outsize;
};

struct layer *
layer_make_full (struct network *net,
                 enum activation_type activation,
                 int width, int height, int depth)
{
    struct fully_layer *fully;
    struct layer *base, *prev;
    int insize, outsize;

    fully = g_new0 (struct fully_layer, 1);
    base = (struct layer *) fully;
    prev = network_layer_last (net);

    insize = prev->width * prev->height * prev->depth;
    outsize = width * height * depth;

    base->net = net;
    base->prev = prev;
    base->type = LAYER_FULLY;
    base->value_v = g_new (float, outsize);
    base->gradient_v = g_new (float, outsize);
    base->weight_v = g_new (float, (insize + 1) * outsize);
    base->delta_v = g_new (float, (insize + 1) * outsize);
    base->value_c = outsize;
    base->weight_c = (insize + 1) * outsize;
    base->width = width;
    base->height = height;
    base->depth = depth;
    base->activation = activation;
    base->forward = forward;
    base->backward = backward;
    base->release = release;
    base->loss = loss;

    fully->insize = insize;
    fully->outsize = outsize;

    network_push_layer (net, base);

    return base;
}

static void
forward (struct layer *lay)
{
    struct fully_layer *fully;
    const float *input_p, *input_e, *weight_p;
    float sum, *value_p, *value_e;

    fully = (struct fully_layer *) lay;
    g_assert (fully->insize == lay->net->input_c);

    weight_p = lay->weight_v;

    value_p = lay->value_v;
    value_e = value_p + lay->value_c;

    input_e = lay->net->input_v + lay->net->input_c;

    while (value_p < value_e) {
        input_p = lay->net->input_v;

        sum = *weight_p++;

        while (input_p < input_e) {
            sum += *weight_p++ * *input_p++;
        }

        *value_p++ = activation_value (lay->activation, sum);
    }

    g_assert (weight_p == lay->weight_v + lay->weight_c);

    network_set_data (lay->net, lay->value_v, lay->value_c);
}

static void
backward (struct layer *lay)
{
    float grad, der, *delta_p, *weight_p;
    int i, j;

    weight_p = lay->weight_v;
    delta_p = lay->delta_v;

    for (j = 0; j < lay->prev->value_c; j++) {
        lay->prev->gradient_v[j] = 0;
    }

    for (i = 0; i < lay->value_c; i++) {
        /* bias */
        grad = lay->gradient_v[i];

        *delta_p = *delta_p * 0.99f + grad * 0.001f;
        *weight_p += *delta_p;

        delta_p++;
        weight_p++;

        for (j = 0; j < lay->prev->value_c; j++) {
            *delta_p = *delta_p * 0.99f + grad * 0.001f * lay->prev->value_v[j];
            *weight_p += *delta_p;

            lay->prev->gradient_v[j] += grad * *weight_p;

            weight_p++;
            delta_p++;
        }
    }

    for (j = 0; j < lay->prev->value_c; j++) {
        der = activation_derivative (lay->prev->activation,
                                     lay->prev->value_v[j]);
        lay->prev->gradient_v[j] *= der;
    }
}

static void
release (struct layer *lay)
{
}

static void
loss (struct layer *lay)
{
    float sum, sub;
    int i;

    g_assert (lay->value_c == lay->net->truth_c);

    sum = 0;

    for (i = 0; i < lay->value_c; i++) {
        sub = lay->net->truth_v[i] - lay->value_v[i];
        sum += sub * sub;

        lay->gradient_v[i] = sub;
    }

    g_assert (sum == sum);
    lay->net->loss = sqrtf (sum);

    for (i = 0; i < lay->value_c; i++) {
        lay->gradient_v[i] *= lay->net->loss;
    }
}
