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
 * along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <glib.h>

struct layer;
struct context;

struct network
{
    struct context *ctx;
    GPtrArray *layers;
    float loss;
    float rate;
    float momentum;
    float decay;
};

struct network *network_make_empty (struct context *ctx);
void network_free (struct network *net);
struct layer *network_layer (struct network *net, int index);
struct layer *network_layer_last (struct network *net);
int network_layer_count (struct network *net);
void network_push_layer (struct network *net, struct layer *lay);
void network_forward (struct network *net);
void network_backward (struct network *net);
void network_compile (struct network *net);
