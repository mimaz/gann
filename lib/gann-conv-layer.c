/*
 * gann-conv-layer.c
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

#include "gann-conv-layer.h"
#include "gann-layer-private.h"
#include "gann-network.h"

struct _GannConvLayer
{
    GannLayer parent_instance;

    gint size;
    gint stride;
};

G_DEFINE_TYPE (GannConvLayer, gann_conv_layer, GANN_TYPE_LAYER);

enum
{
    PROP_0,
    PROP_SIZE,
    PROP_STRIDE,
    PROP_FILTERS,
    N_PROPS,
};

static GParamSpec *props[N_PROPS];

static void
constructed (GObject *gobj);

static void
gann_conv_layer_init (GannConvLayer *self)
{
    g_object_bind_property (self, "depth", self, "filters",
                            G_BINDING_SYNC_CREATE);
}

static void
gann_conv_layer_class_init (GannConvLayerClass *cls)
{
    GObjectClass *gcls = G_OBJECT_CLASS (cls);

    gcls->constructed = constructed;

    props[PROP_SIZE] =
        g_param_spec_int ("size",
                          "Size",
                          "Kernel size",
                          1, G_MAXINT32, 3,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS);

    props[PROP_STRIDE] =
        g_param_spec_int ("stride",
                          "Stride",
                          "Kernel stride",
                          1, G_MAXINT32, 1,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS);

    g_object_class_install_properties (gcls, N_PROPS, props);
}

static void
constructed (GObject *gobj)
{
    GannLayer *layer;
    GannConvLayer *self;
    GannNetwork *network;
    struct layer *core;

    layer = GANN_LAYER (gobj);
    self = GANN_CONV_LAYER (gobj);
    network = gann_layer_get_network (layer);

    core = layer_make_conv (gann_network_get_core (network),
                            gann_conv_layer_get_size (self),
                            gann_layer_get_depth (layer),
                            gann_conv_layer_get_stride (self),
                            gann_layer_get_activation (layer),
                            NULL);

    gann_layer_set_core (layer, core);

    G_OBJECT_CLASS (gann_conv_layer_parent_class)->constructed (gobj);
}

gint
gann_conv_layer_get_size (GannConvLayer *self)
{
    return self->size;
}

gint
gann_conv_layer_get_stride (GannConvLayer *self)
{
    return self->stride;
}