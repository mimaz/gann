#include "network.h"
#include "layer.h"

struct network *
network_make_empty ()
{
    struct network *net;

    net = g_new0 (struct network, 1);
    net->layers = g_ptr_array_new ();

    return net;
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
network_set_data (struct network *net, float *values, int size)
{
    net->data_v = values;
    net->data_c = size;
}

void
network_set_truth (struct network *net, float *values, int size)
{
    net->truth_v = values;
    net->truth_c = size;
}

void
network_forward (struct network *net)
{
    int i, count;
    struct layer *lay;

    count = network_layer_count (net);

    for (i = 0; i < count; i++) {
        g_message ("forward %d", i);
        lay = network_layer (net, i);
        net->input_v = net->data_v;
        net->input_c = net->data_c;
        layer_forward (lay);
    }
}

void
network_backward (struct network *net)
{
    struct layer *lay;

    lay = network_layer_last (net);
    g_assert (lay->loss != NULL);
    lay->loss (lay);

    g_message ("loss: %f", net->loss);
}

void
network_randomize (struct network *net)
{
    int count, i;

    count = network_layer_count (net);

    for (i = 0; i < count; i++) {
        layer_randomize (network_layer (net, i));
    }
}
