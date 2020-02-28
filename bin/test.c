#include "network.h"
#include "layer.h"

gint
main (gint argc, gchar **argv)
{
    struct network *net;
    int p, q;
    float input[2], output[4], diff, avg;

    net = network_make_empty ();

    layer_make_input (net, 1, 1, 2);
    layer_make_full (net, ACTIVATION_LEAKY, 1, 1, 10);
    layer_make_full (net, ACTIVATION_ELU, 1, 1, 10);
    layer_make_full (net, ACTIVATION_LEAKY, 1, 1, 10);
    layer_make_full (net, ACTIVATION_ELU, 1, 1, 10);
    layer_make_full (net, ACTIVATION_LEAKY, 1, 1, 10);
    layer_make_full (net, ACTIVATION_ELU, 1, 1, 10);
    layer_make_full (net, ACTIVATION_STEP, 1, 1, 4);

    network_randomize (net);

    avg = -1;

    for (int i = 0; i < 5000000; i++) {
        p = rand () & 1;
        q = rand () & 1;

        input[0] = p;
        input[1] = q;
        output[0] = p ^ q;
        output[1] = p & q;
        output[2] = p | q;
        output[3] = (p | q) == 0;

        network_set_data (net, input, 2);
        network_set_truth (net, output, 4);
        network_forward (net);
        network_backward (net);

        diff = 0;
        for (int i = 0; i < 4; i++)
            diff += fabsf (net->data_v[i] - output[i]);

        if (avg < 0) {
            avg = diff;
        } else {
            avg = avg * 0.99f + diff * 0.01f;
        }

        g_message ("%d: %d ^ %d = [%f, %f, %f, %f] :: [%f, %f, %f, %f] :: %.02f",
                   i, p, q, output[0], output[1], output[2], output[3],
                   net->data_v[0], net->data_v[1], net->data_v[2], net->data_v[3],
                   diff);

        if (avg < 0.01)
            break;
    }

    return 0;
}
