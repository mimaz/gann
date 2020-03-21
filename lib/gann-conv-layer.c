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
#include "gann-network.h"

#include "core/core.h"

struct _GannConvLayer
{
    GannLayer parent_instance;

    gint kernel_size;
    gint stride;
    gint filters;
    gfloat *filterdata;
    gsize filtersize;
};

G_DEFINE_TYPE (GannConvLayer, gann_conv_layer, GANN_TYPE_LAYER);

enum
{
    PROP_0,
    PROP_KERNEL_SIZE,
    PROP_STRIDE,
    PROP_FILTERS,
    N_PROPS,
};

static GParamSpec *props[N_PROPS];

static void set_property (GObject *gobj, guint propid,
                          const GValue *value, GParamSpec *spec);
static void get_property (GObject *gobj, guint propid,
                          GValue *value, GParamSpec *spec);
static void finalize (GObject *gobj);
static void constructed (GObject *gobj);

static void
gann_conv_layer_init (GannConvLayer *self)
{
    g_object_bind_property (self, "depth",
                            self, "filters",
                            G_BINDING_BIDIRECTIONAL |
                            G_BINDING_SYNC_CREATE);
}

static void
gann_conv_layer_class_init (GannConvLayerClass *cls)
{
    GObjectClass *gcls = G_OBJECT_CLASS (cls);

    gcls->set_property = set_property;
    gcls->get_property = get_property;
    gcls->finalize = finalize;
    gcls->constructed = constructed;

    props[PROP_KERNEL_SIZE] =
        g_param_spec_int ("kernel-size",
                          "Kernel size",
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

    props[PROP_FILTERS] =
        g_param_spec_int ("filters",
                          "Filters",
                          "Number of filters",
                          1, G_MAXINT32, 1,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT |
                          G_PARAM_STATIC_STRINGS);

    g_object_class_install_properties (gcls, N_PROPS, props);
}

static void
set_property (GObject *gobj,
              guint propid,
              const GValue *value,
              GParamSpec *spec)
{
    GannConvLayer *self = GANN_CONV_LAYER (gobj);

    switch (propid) {
    case PROP_KERNEL_SIZE:
        self->kernel_size = g_value_get_int (value);
        break;

    case PROP_STRIDE:
        self->stride = g_value_get_int (value);
        break;

    case PROP_FILTERS:
        self->filters = g_value_get_int (value);
        break;

    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (gobj, propid, spec);
    }
}

static void
get_property (GObject *gobj,
              guint propid,
              GValue *value,
              GParamSpec *spec)
{
    GannConvLayer *self = GANN_CONV_LAYER (gobj);

    switch (propid) {
    case PROP_KERNEL_SIZE:
        g_value_set_int (value, self->kernel_size);
        break;

    case PROP_STRIDE:
        g_value_set_int (value, self->stride);
        break;

    case PROP_FILTERS:
        g_value_set_int (value, self->filters);
        break;

    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (gobj, propid, spec);
    }
}

static void
finalize (GObject *gobj)
{
    G_OBJECT_CLASS (gann_conv_layer_parent_class)->finalize (gobj);
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

    self->filtersize = self->kernel_size * self->kernel_size
                     * gann_layer_get_depth (layer)
                     * self->filters;
    self->filterdata = g_new (gfloat, self->filtersize);

    core = layer_make_conv (gann_network_get_core (network),
                            self->kernel_size,
                            self->stride,
                            self->filters,
                            gann_layer_get_activation (layer),
                            NULL);

    gann_layer_set_core (layer, core);
}

GannConvLayer *
gann_conv_layer_new (GannNetwork *network,
                     gint kernel_size,
                     gint stride,
                     gint filters,
                     const gchar *activation)
{
    return g_object_new (GANN_TYPE_CONV_LAYER,
                         "network", network,
                         "kernel-size", kernel_size,
                         "stride", stride,
                         "filters", filters,
                         "width", -1,
                         "height", -1,
                         "depth", filters,
                         "activation", activation,
                         NULL);
}

gint
gann_conv_layer_get_kernel_size (GannConvLayer *self)
{
    return self->kernel_size;
}

gint
gann_conv_layer_get_stride (GannConvLayer *self)
{
    return self->stride;
}

gint
gann_conv_layer_get_filters (GannConvLayer *self)
{
    return self->filters;
}
