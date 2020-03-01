#include "network.h"
#include "layer.h"

struct network *
network_make_empty ()
{
    struct network *net;

    net = g_new0 (struct network, 1);
    net->layers = g_ptr_array_new ();
    net->rate = 0.001f;
    net->momentum = 0.99f;
    net->decay = 0.00001f;

    return net;
}

void
network_free (struct network *net)
{
    struct layer *lay;
    int count, i;

    count = network_layer_count (net);

    for (i = 0; i < count; i++) {
        lay = network_layer (net, i);
        layer_free (lay);
    }
}

struct layer *
network_layer (struct network *net, int index)
{
    int count;
    count = network_layer_count (net);
    if (index < 0) {
        index += count;
        g_assert (index >= 0);
        return network_layer (net, index);
    }
    g_assert (index < count);
    return g_ptr_array_index (net->layers, index);
}

struct layer *
network_layer_last (struct network *net)
{
    return network_layer (net, -1);
}

int
network_layer_count (struct network *net)
{
    return net->layers->len;
}

void
network_push_layer (struct network *net, struct layer *lay)
{
    g_ptr_array_insert (net->layers, -1, lay);
}

void
network_forward (struct network *net)
{
    int i, count;
    struct layer *lay;

    count = network_layer_count (net);

    for (i = 0; i < count; i++) {
        lay = network_layer (net, i);
        g_assert (lay->forward != NULL);
        lay->forward (lay);
    }
}

void
network_backward (struct network *net)
{
    struct layer *lay;
    int i, j, count;

    count = network_layer_count (net);

    for (i = 0; i < count; i++) {
        lay = network_layer (net, i);
        for (j = 0; j < lay->size; j++) {
            lay->gradient_v[j] = 0;
        }
    }

    net->loss = 0;

    for (i = count; i > 0; i--) {
        lay = network_layer (net, i - 1);
        g_assert (lay->backward != NULL);
        lay->backward (lay);
    }
}

void
network_randomize (struct network *net)
{
    struct layer *lay;
    int count, i;

    count = network_layer_count (net);

    for (i = 0; i < count; i++) {
        lay = network_layer (net, i);
        if (lay->initialize != NULL) {
            lay->initialize (lay);
        }
    }
}
