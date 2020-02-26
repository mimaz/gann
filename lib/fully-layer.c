#include "layer.h"
#include "network.h"

static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);

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
    base->type = LAYER_FULLY;
    base->weight_v = g_new (float, (insize + 1) * outsize);
    base->value_v = g_new (float, outsize);
    base->weight_c = insize * outsize;
    base->value_c = outsize;
    base->width = width;
    base->height = height;
    base->depth = depth;
    base->activation = activation;
    base->forward = forward;
    base->backward = backward;
    base->release = release;

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

    weight_p = lay->weight_v;

    value_p = lay->value_v;
    value_e = value_p + lay->value_c;

    input_e = lay->net->input_v + lay->net->input_c;

    g_assert (fully->insize == lay->net->input_c);

    while (value_p < value_e) {
        input_p = lay->net->input_v;

        sum = *weight_p++;

        while (input_p < input_e) {
            sum += *weight_p++ * *input_p++;
        }

        *value_p++ = sum;
    }

    layer_activate (lay);
    network_set_data (lay->net, lay->value_v, lay->value_c);
}

static void 
backward (struct layer *lay)
{
}

static void 
release (struct layer *lay)
{
}
