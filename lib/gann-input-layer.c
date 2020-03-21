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
 * along with Gann.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "gann-input-layer.h"
#include "gann-network.h"

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
    GannNetwork *network;
    GannLayer *layer;
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
}

GannInputLayer *
gann_input_layer_new (GannNetwork *network,
                      gint width,
                      gint height,
                      gint depth)
{
    return g_object_new (GANN_TYPE_INPUT_LAYER,
                         "network", network,
                         "width", width,
                         "height", height,
                         "depth", depth,
                         NULL);
}

void
gann_input_layer_set_data (GannInputLayer *self,
                           const gfloat *data,
                           gsize datasize)
{
    layer_input_set_data (gann_layer_get_core (GANN_LAYER (self)),
                          data, datasize);
}

void
gann_input_layer_set_data_bytes (GannInputLayer *self,
                                 const guint8 *data,
                                 gsize datasize)
{
    GannLayer *layer = GANN_LAYER (self);
    gint width, height, depth;
    gfloat *floats;
    gsize i;

    width = gann_layer_get_width (layer);
    height = gann_layer_get_height (layer);
    depth = gann_layer_get_depth (layer);

    g_assert (datasize == width * height * depth);

    floats = g_newa (gfloat, width * height * depth);

    for (i = 0; i < datasize; i++) {
        floats[i] = data[i] / 255.0f;
    }

    gann_input_layer_set_data (self, floats, datasize);
}
