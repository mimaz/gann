/*
 * gann-dense-layer.c
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

#include "gann-dense-layer.h"
#include "gann-network.h"
#include "gann-layer-private.h"

#include "core/core.h"

struct _GannDenseLayer
{
    GannLayer parent_instance;
};

G_DEFINE_TYPE (GannDenseLayer, gann_dense_layer, GANN_TYPE_LAYER);

static void attached (GannLayer *layer);

static void
gann_dense_layer_init (GannDenseLayer *self)
{
}

static void
gann_dense_layer_class_init (GannDenseLayerClass *cls)
{
    GannLayerClass *lcls = GANN_LAYER_CLASS (cls);

    lcls->attached = attached;
}

static void
attached (GannLayer *layer)
{
    GannNetwork *network;
    struct layer *core;

    network = gann_layer_get_network (layer);

    core = layer_make_dense (gann_network_get_core (network),
                             gann_layer_get_width (layer),
                             gann_layer_get_height (layer),
                             gann_layer_get_depth (layer),
                             gann_layer_get_activation (layer),
                             NULL);
    gann_layer_set_core (layer, core);
}

GannDenseLayer *
gann_dense_layer_new (gint width,
                      gint height,
                      gint depth,
                      const gchar *activation)
{
    return g_object_new (GANN_TYPE_DENSE_LAYER,
                         "width", width,
                         "height", height,
                         "depth", depth,
                         "activation", activation,
                         NULL);
}
