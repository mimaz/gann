/*
 * network.h
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

#pragma once

#include <glib.h>

#define NETWORK_FLAG_BACKPROP 1

struct layer;
struct context;

struct network
{
    /* context pointer */
    struct context *ctx;

    /* array of layer pointers */
    GPtrArray *layers;

    /* some flags */
    int flags;

    /* lates error loss */
    float loss;

    /* learning parameters */
    float rate;
    float momentum;
    float decay;
};

/*
 * network_create
 * Creates new network instance for given context
 * ctx: context pointer
 */
struct network *network_create (struct context *ctx);

/*
 * network_free
 * Frees the network
 */
void network_free (struct network *net);

/*
 * network_layer:
 * Gives pointer to nth layer
 * index: index of the layer
 * returns: pointer to the layer
 */
struct layer *network_layer (struct network *net,
                             int index);

/*
 * network_layer_last:
 * returns: pointer to the lastly added layer
 */
struct layer *network_layer_last (struct network *net);

/*
 * network_layer_count:
 * returns: number of added layers
 */
int network_layer_count (struct network *net);

/*
 * network_push_layer:
 * Adds layer to the layer list. After call the layer is
 * owned by the network.
 * lay: pointer to the layer
 */
void network_push_layer (struct network *net, struct layer *lay);

/*
 * network_forward:
 * Propagates network forward
 */
void network_forward (struct network *net);

/*
 * network_backward:
 * Backpropagates error
 */
void network_backward (struct network *net);
