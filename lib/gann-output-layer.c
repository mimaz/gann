/*
 * gann-output-layer.c
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

#include "gann-output-layer.h"

#include "gann-network.h"
#include "gann-layer-private.h"

#include "core/layer.h"

struct _GannOutputLayer
{
    GannLayer parent_instance;
};

G_DEFINE_TYPE (GannOutputLayer, gann_output_layer, GANN_TYPE_LAYER);

static void attached (GannLayer *layer);

static void
gann_output_layer_init (GannOutputLayer *self)
{
}

static void
gann_output_layer_class_init (GannOutputLayerClass *cls)
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

    g_assert (gann_layer_get_width (layer) == 0);
    g_assert (gann_layer_get_height (layer) == 0);
    g_assert (gann_layer_get_depth (layer) == 0);

    core = layer_make_output (gann_network_get_core (network));
    gann_layer_set_core (layer, core);

    g_object_set (layer,
                  "width", core->width,
                  "height", core->height,
                  "depth", core->depth,
                  NULL);
}

GannOutputLayer *
gann_output_layer_new ()
{
    return g_object_new (GANN_TYPE_OUTPUT_LAYER,
                         NULL);
}

void
gann_output_layer_set_truth (GannOutputLayer *self,
                             const gfloat *data,
                             gsize datasize)
{
    layer_output_set_truth (gann_layer_get_core (GANN_LAYER (self)),
                            data, datasize);
}
