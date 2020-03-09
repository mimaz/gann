/*
 * gann-input-layer.c
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

#include "gann-input-layer.h"
#include "gann-network.h"
#include "gann-layer-private.h"

#include "core/core.h"

struct _GannInputLayer
{
    GObject parent_instance;
};

G_DEFINE_TYPE (GannInputLayer, gann_input_layer,
               GANN_TYPE_LAYER);

static void dispose (GObject *gobj);
static void constructed (GObject *gobj);

static void
gann_input_layer_init (GannInputLayer *self)
{
}

static void
gann_input_layer_class_init (GannInputLayerClass *cls)
{
    GObjectClass *gcls = G_OBJECT_CLASS (cls);

    gcls->dispose = dispose;
    gcls->constructed = constructed;
}

static void
dispose (GObject *gobj)
{
    /* allocated data block is freed by core layer */
    G_OBJECT_CLASS (gann_input_layer_parent_class)->dispose (gobj);
}

static void
constructed (GObject *gobj)
{
    GannLayer *layer;
    GannNetwork *network;
    gint width, height, depth;
    struct layer *core;

    layer = GANN_LAYER (gobj);
    network = gann_layer_get_network (layer);

    width = gann_layer_get_width (layer);
    height = gann_layer_get_height (layer);
    depth = gann_layer_get_depth (layer);

    core = layer_make_input (gann_network_get_core (network),
                             width, height, depth);
    gann_layer_set_core (layer, core);

    G_OBJECT_CLASS (gann_input_layer_parent_class)->constructed (gobj);
}

void
gann_input_layer_set_input (GannInputLayer *self,
                            const gfloat *data,
                            gsize datasize)
{
    layer_input_set_data (gann_layer_get_core (GANN_LAYER (self)),
                          data, datasize);
}

void
gann_input_layer_set_input_floats (GannInputLayer *self,
                                   gfloat first, ...)
{
    GArray *arr;
    va_list args;

    arr = g_array_new (FALSE, FALSE, sizeof (gfloat));
    va_start (args, first);

    do {
        g_array_append_val (arr, first);

        /* floats are promoted as doubles */
        first = va_arg (args, gdouble);
    } while (first >= 0);

    gann_input_layer_set_input (self, (gfloat *) arr->data, arr->len);

    g_array_unref (arr);
    va_end (args);
}
