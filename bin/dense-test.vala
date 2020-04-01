/*
 * dense-test.vala
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

public void dense_test () throws Error
{
    /*
     * Create context and network instances
     */
    var context = new Gann.Context ();
    var network = new Gann.Network (context, 0.5f, 0.99f, 1.0f);

    /*
     * Create layers, keep references to input in output ones
     *
     * numeric arguments are tensor size width x height x depth
     * available activations: relu, leaky, sigmoid, softplus, linear
     */
    var input = network.create_input (1, 1, 2);
    network.create_dense (1, 1, 64, "relu");
    network.create_dense (1, 1, 1, "linear");
    var output = network.create_output ();

    /*
     * Rand instance used for generating test data
     */
    var rand = new Rand.with_seed (0);

    for (var i = 0; i < 1000000; i++) {
        /*
         * For example purpose here we randomize two bits 
         * and make simple logical operation to calculate expected
         * output. Then convert these values to proper float arrays
         * for both input and output layer (so input and truth values)
         */
        int p = (int) rand.boolean ();
        int q = (int) rand.boolean ();
        int e = (int) (p == 0 && q == 0);

        float[] indata = { p, q, };
        float[] outdata = { e, };

        /*
         * Feed input data and run forward propagation
         */
        input.set_data (indata);
        network.forward ();

        /*
         * Read calculated value
         */
        var r = output.get_data ()[0];

        /*
         * Set truth value and run backpropagation
         * Here the network is learning how to
         * output correct values
         */
        output.set_truth (outdata);
        network.backward ();

        /*
         * Get average loss from last sessions
         */
        var loss = network.get_average_loss ();

        /*
         * Print session summary
         */
        message ("epoch %d; input %d, %d; expected %d; answer %f; loss %f",
                 i, p, q, e, r, loss);

        /*
         * Break if trained enough
         */
        if (i > 10 && loss < 0.05f) {
            break;
        }

        /*
         * handle loss error
         */
        if (loss != loss) {
            error ("got nan loss, it may be caused by too fast weight " +
                   "changes, badly initialized weights, invalid network " +
                   "structure or something else.");
        }
    }
}
