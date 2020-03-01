#include "gann.h"

gint
main (gint argc, gchar **argv)
{
    GannInputLayer *in;
    GannOutputLayer *out;
    GannContext *context;
    GannNetwork *net;
    gint i, p, q, r;

    context = gann_context_new ();
    net = gann_network_new_full (context, 0.01f, 0.9f, 0.999999f);
    in = gann_network_create_input (net, 1, 1, 2);
    gann_network_create_fully (net, 1, 1, 3, GANN_ACTIVATION_RELU);
    gann_network_create_fully (net, 1, 1, 1, GANN_ACTIVATION_STEP);
    out = gann_network_create_output (net);

    for (i = 0; i < 500000; i++) {
        p = rand () & 1;
        q = rand () & 1;
        r = p == 0 || q == 1;

        gann_input_layer_set_input_floats (in, (gfloat) p, (gfloat) q, -1.0f);
        gann_output_layer_set_truth_floats (out, (gfloat) r, -1.0f);
        gann_network_forward (net);
        gann_network_backward (net);
        g_message ("%d result: %f %d %f", i,
                   gann_network_get_average_loss (net), r,
                   gann_layer_get_data (GANN_LAYER (out), NULL)[0]);

        if (gann_network_get_average_loss (net) < 0.01 && i > 10) {
            break;
        }
    }

    g_object_unref (context);
    g_object_unref (net);
}
