#include "network.h"
#include "layer.h"

gint
main (gint argc, gchar **argv)
{
    struct network *net;
    float input[2], output[1];

    net = network_make_empty ();
    g_message ("ok");
    layer_make_input (net, 1, 1, 2);
    g_message ("ok");
    layer_make_full (net, ACTIVATION_SIGMOID, 1, 1, 3);
    g_message ("ok");
    layer_make_full (net, ACTIVATION_SIGMOID, 1, 1, 1);
    g_message ("ok");

    network_randomize (net);

    input[0] = 1;
    input[1] = 0;
    output[0] = 1;

    network_set_data (net, input, 2);
    network_set_truth (net, output, 1);
    network_forward (net);
    network_backward (net);

    g_message ("data: %d: %f", net->data_c, net->data_v[0]);

    return 0;
}
