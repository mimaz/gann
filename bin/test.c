/*
 * test.c
 *
 * Copyright 2020 Mieszko Mazurek <mimaz@gmx.com>
 *
 * This file is part of Gann.
 *
 * Gann is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Gann is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Gann.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "gann.h"

gint
main (gint argc, gchar **argv)
{
    GannInputLayer *in;
    GannOutputLayer *out;
    GannContext *context;
    GannNetwork *net;
    gint i, p, q, e;
    gfloat r, loss;

    /*
     * Create context
     */
    context = gann_context_new ();

    /*
     * Create network with given learning rate, momentum and weight decay
     */
    net = gann_network_new_full (context, 0.5f, 0.9f, 1.0f);

    /*
     * Create input layer with shape 1x1x2
     */
    in = gann_network_create_input (net, 1, 1, 2);

    /*
     * Create some hidden layers
     * The network is way over complex for such simple learning data set
     * But here it's just for the example purpose
     */
    gann_network_create_dense (net, 1, 1, 8, "leaky");
    gann_network_create_dense (net, 1, 1, 4, "softplus");
    gann_network_create_dense (net, 1, 1, 1, "sigmoid");

    /*
     * Create output layer
     * It's shape will be same as last hidden layer's one
     */
    out = gann_network_create_output (net);

    for (i = 0; i < 10000; i++) {
        /*
         * Randomize two bits and make simple operation, in this case XOR
         */
        p = rand () & 1;
        q = rand () & 1;
        e = p ^ q;

        /*
         * Put $p and $q variables as input
         * -1 is a list guard
         */
        gann_input_layer_set_input_floats (in, (gfloat) p, (gfloat) q, -1.0f);

        /*
         * Propagate input data forward
         * Here network calculates it's output value
         */
        gann_network_forward (net);

        /*
         * Read calculated output
         */
        r = gann_layer_get_data (GANN_LAYER (out), NULL)[0];

        /*
         * Set expected result (truth) for given input
         * -1 is a list guard
         */
        gann_output_layer_set_truth_floats (out, (gfloat) e, -1.0f);

        /*
         * Propagate error back
         * Here the network is learning how to output expected result
         */
        gann_network_backward (net);

        /*
         * Get average error from last few training sessions
         */
        loss = gann_network_get_average_loss (net);

        /*
         * Print session summary
         */
        g_message ("%d: result: %f, expected: %d, loss %f",
                   i, r, e, loss);

        /*
         * Break if the error is small enough
         */
        if (loss < 0.1f && i > 10 || loss != loss) {
            break;
        }
    }

    /*
     * Clear out objects
     */
    g_object_unref (net);
    g_object_unref (context);
}
