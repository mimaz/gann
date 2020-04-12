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

struct _GannConvLayer
{
    GannLayer parent_instance;

    gint kernel_width;
    gint kernel_height;
    gint kernel_stride;

    gint xshift;
    gint yshift;

    gfloat *filterdata;
    gsize filtersize;
};

G_DEFINE_TYPE (GannConvLayer, gann_conv_layer, GANN_TYPE_LAYER);

enum
{
    PROP_0,
    PROP_KERNEL_WIDTH,
    PROP_KERNEL_HEIGHT,
    PROP_KERNEL_STRIDE,
    N_PROPS,
};

static GParamSpec *props[N_PROPS];

static void set_property (GObject *gobj, guint propid,
                          const GValue *value, GParamSpec *spec);
static void get_property (GObject *gobj, guint propid,
                          GValue *value, GParamSpec *spec);
static void finalize (GObject *gobj);
static void constructed (GObject *gobj);
static void compile (GannLayer *layer);

static void
gann_conv_layer_init (GannConvLayer *self)
{
}

static void
gann_conv_layer_class_init (GannConvLayerClass *cls)
{
    GannLayerClass *lcls = GANN_LAYER_CLASS (cls);
    GObjectClass *gcls = G_OBJECT_CLASS (cls);

    gcls->set_property = set_property;
    gcls->get_property = get_property;
    gcls->finalize = finalize;
    gcls->constructed = constructed;

    lcls->compile = compile;

    props[PROP_KERNEL_WIDTH] =
        g_param_spec_int ("kernel-width",
                          "Kernel width",
                          "Kernel width",
                          1, G_MAXINT32, 1,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS);

    props[PROP_KERNEL_HEIGHT] =
        g_param_spec_int ("kernel-height",
                          "Kernel height",
                          "Kernel height",
                          1, G_MAXINT32, 1,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS);

    props[PROP_KERNEL_STRIDE] =
        g_param_spec_int ("kernel-stride",
                          "Kernel stride",
                          "Kernel stride",
                          1, G_MAXINT32, 1,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
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
    case PROP_KERNEL_WIDTH:
        self->kernel_width = g_value_get_int (value);
        break;

    case PROP_KERNEL_HEIGHT:
        self->kernel_height = g_value_get_int (value);
        break;

    case PROP_KERNEL_STRIDE:
        self->kernel_stride = g_value_get_int (value);
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
    case PROP_KERNEL_WIDTH:
        g_value_set_int (value, self->kernel_width);
        break;

    case PROP_KERNEL_HEIGHT:
        g_value_set_int (value, self->kernel_height);
        break;

    case PROP_KERNEL_STRIDE:
        g_value_set_int (value, self->kernel_stride);
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
    G_OBJECT_CLASS (gann_conv_layer_parent_class)->constructed (gobj);
}

static void
compile (GannLayer *layer)
{
    GannConvLayer *self;
    GannLayer *prev;

    self = GANN_CONV_LAYER (layer);
    prev = gann_layer_prev_layer (layer);
    g_assert_nonnull (prev);

    g_message ("compiled conv");
    self->filtersize = self->kernel_width * self->kernel_height
                     * gann_layer_get_depth (layer)
                     * gann_layer_get_depth (prev);
    self->filterdata = g_new (gfloat, self->filtersize);

    GANN_LAYER_CLASS (gann_conv_layer_parent_class)->compile (layer);
}

/**
 * gann_conv_layer_new:
 * @network: network instance to attach to
 * @kernel_size: size of kernel, N for NxN kernel
 * @stride: kernel stride (usually 1)
 * @filters: number of output filters
 * @activation: activation function name
 */
GannConvLayer *
gann_conv_layer_new (GannNetwork *network,
                     gint kernel_size,
                     gint stride,
                     gint filters,
                     const gchar *activation)
{
    return g_object_new (GANN_TYPE_CONV_LAYER,
                         "network", network,
                         "kernel-width", kernel_size,
                         "kernel-height", kernel_size,
                         "kernel-stride", stride,
                         "width", -1,
                         "height", -1,
                         "depth", filters,
                         "activation", activation,
                         NULL);
}

/**
 * gann_conv_layer_get_kernel_width:
 *
 * returns: kernel width
 */
gint
gann_conv_layer_get_kernel_width (GannConvLayer *self)
{
    return self->kernel_width;
}

/**
 * gann_conv_layer_get_kernel_height:
 *
 * returns: kernel height
 */
gint
gann_conv_layer_get_kernel_height (GannConvLayer *self)
{
    return self->kernel_height;
}

/**
 * gann_conv_layer_get_stride:
 *
 * returns: stride
 */
gint
gann_conv_layer_get_kernel_stride (GannConvLayer *self)
{
    return self->kernel_stride;
}
