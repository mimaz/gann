/*
 * network.c
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

#include "network.h"
#include "layer.h"
#include "context.h"

#include <math.h>

struct network *
network_create (struct context *ctx)
{
    struct network *net;

    net = g_new0 (struct network, 1);
    net->ctx = ctx;
    net->layers = g_ptr_array_new_with_free_func ((GDestroyNotify)
                                                  layer_free);
    net->flags = NETWORK_FLAG_BACKPROP;
    net->loss = 0;
    net->rate = 0.5f;
    net->momentum = 0.9f;
    net->decay = 1.0f;

    /* manually add itself to the context */
    ctx->netlist = g_slist_prepend (ctx->netlist, net);

    return net;
}

void
network_free (struct network *net)
{
    /* manually remove itself from the context */
    net->ctx->netlist = g_slist_remove (net->ctx->netlist, net);

    g_ptr_array_unref (net->layers);
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

int
network_layer_count (struct network *net)
{
    return net->layers->len;
}

void
network_backward (struct network *net)
{
    /* struct layer *lay; */
    /* int i, count; */
    /*  */
    /* count = network_layer_count (net); */
    /*  */
    /* for (i = 0; i < count; i++) { */
    /*     lay = network_layer (net, i); */
    /*     layer_clear_gradient (lay); */
    /* } */
    /*  */
    /* for (i = count; i > 0; i--) { */
    /*     lay = network_layer (net, i - 1); */
    /*     layer_backward (lay); */
    /* } */
    /*  */
    /* net->loss = 0; */
    /*  */
    /* for (i = 0; i < count; i++) { */
    /*     lay = network_layer (net, i - 1); */
    /*     net->loss += lay->loss; */
    /* } */
    /*  */
    /* net->loss = logf (net->loss + 1); */
}
