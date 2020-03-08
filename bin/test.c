#include "gann.h"

gint
main (gint argc, gchar **argv)
{
    GannInputLayer *in;
    GannOutputLayer *out;
    GannContext *context;
    GannNetwork *net;
    gint i, p, q, r;
    gfloat avg;

    srand(0);
    context = gann_context_new ();
    net = gann_network_new_full (context, 0.8f, 0.9f, 1.0f);
    in = gann_network_create_input (net, 1, 1, 2);
    gann_network_create_dense (net, 1, 1, 20, GANN_ACTIVATION_SIGMOID);
    gann_network_create_dense (net, 1, 1, 20, GANN_ACTIVATION_SIGMOID);
    gann_network_create_dense (net, 1, 1, 1, GANN_ACTIVATION_SIGMOID);
    out = gann_network_create_output (net);

    g_object_unref (context);

    for (i = 0; i < 100000; i++) {
        p = rand () & 1;
        q = rand () & 1;
        r = p ^ q;

        gann_input_layer_set_input_floats (in, (gfloat) p, (gfloat) q, -1.0f);
        gann_output_layer_set_truth_floats (out, (gfloat) r, -1.0f);
        gann_network_forward (net);
        gann_network_backward (net);
        avg = gann_network_get_average_loss (net);
        g_assert (avg == avg);
        if (i % 1 == 0) {
            g_message ("%d result: %f %d %f", i, avg, r,
                    gann_layer_get_data (GANN_LAYER (out), NULL)[0]);
            if (gann_network_get_average_loss (net) < 0.1 && i > 10) {
                break;
            }
        }
    }

    g_object_unref (net);
}
