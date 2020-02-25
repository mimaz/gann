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

    fully->insize = insize;
    fully->outsize = outsize;

    network_push_layer (net, base);

    return base;
}

static void
forward (struct layer *lay)
{
    struct fully_layer *fully;
    int x, y;
    float sum;
    const float *input;

    fully = (struct fully_layer *) lay;
    input = lay->net->input_v;

    g_assert (fully->insize == lay->net->input_c);

    for (y = 0; y < fully->outsize; y++) {
        sum = lay->bias_v[y];

        for (x = 0; x < fully->insize; x++) {
            sum += lay->weight_v[y * fully->outsize + x] * input[x];
        }

        lay->value_v[y] = sum;
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
    g_clear_pointer (&lay->weight_v, g_free);
    g_clear_pointer (&lay->value_v, g_free);
    g_clear_pointer (&lay->bias_v, g_free);
}
