/*
 * gann-layer.c
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

#include "gann-layer.h"

#include "gann-layer-private.h"
#include "gann-input-layer.h"
#include "gann-output-layer.h"
#include "gann-dense-layer.h"
#include "gann-network.h"

#include "core/layer.h"

typedef struct {
    GannNetwork *network;

    gint width;
    gint height;
    gint depth;
    gchar *activation;

    float *value_buff;

    struct layer *l;
} GannLayerPrivate;

G_DEFINE_TYPE_WITH_PRIVATE (GannLayer, gann_layer, G_TYPE_OBJECT);

enum
{
    PROP_0,
    PROP_NETWORK,
    PROP_WIDTH,
    PROP_HEIGHT,
    PROP_DEPTH,
    PROP_ACTIVATION,
    N_PROPS,
};

static GParamSpec *props[N_PROPS];

static void dispose (GObject *gobj);
static void constructed (GObject *gobj);
static void set_property (GObject *gobj, guint propid,
                          const GValue *value, GParamSpec *spec);
static void get_property (GObject *gobj, guint propid,
                          GValue *value, GParamSpec *spec);

static void
gann_layer_init (GannLayer *self)
{
}

static void
gann_layer_class_init (GannLayerClass *cls)
{
    GObjectClass *gcls = G_OBJECT_CLASS (cls);

    gcls->dispose = dispose;
    gcls->constructed = constructed;
    gcls->set_property = set_property;
    gcls->get_property = get_property;

    props[PROP_NETWORK] =
        g_param_spec_object ("network",
                             "Network",
                             "Network object",
                             GANN_TYPE_NETWORK,
                             G_PARAM_READWRITE |
                             G_PARAM_CONSTRUCT_ONLY |
                             G_PARAM_STATIC_STRINGS);

    props[PROP_WIDTH] =
        g_param_spec_int ("width",
                          "Width",
                          "Layer width",
                          0, G_MAXINT16, 0,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT |
                          G_PARAM_STATIC_STRINGS);
    props[PROP_HEIGHT] =
        g_param_spec_int ("height",
                          "Height",
                          "Layer height",
                          0, G_MAXINT16, 0,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT |
                          G_PARAM_STATIC_STRINGS);

    props[PROP_DEPTH] =
        g_param_spec_int ("depth",
                          "Depth",
                          "Layer depth",
                          0, G_MAXINT16, 0,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT |
                          G_PARAM_STATIC_STRINGS);

    props[PROP_ACTIVATION] =
        g_param_spec_string ("activation",
                             "Activation",
                             "Layer activation",
                             "sigmoid",
                             G_PARAM_READWRITE |
                             G_PARAM_CONSTRUCT_ONLY |
                             G_PARAM_STATIC_STRINGS);

    g_object_class_install_properties (gcls, N_PROPS, props);
}

static void
dispose (GObject *gobj)
{
    GannLayer *self = GANN_LAYER (gobj);
    GannLayerPrivate *p = gann_layer_get_instance_private (self);

    g_clear_weak_pointer (&p->network);
    g_clear_pointer (&p->value_buff, g_free);

    G_OBJECT_CLASS (gann_layer_parent_class)->dispose (gobj);
}

static void
constructed (GObject *gobj)
{
    GannLayer *self = GANN_LAYER (gobj);
    GannLayerPrivate *p = gann_layer_get_instance_private (self);

    p->value_buff = NULL;

    G_OBJECT_CLASS (gann_layer_parent_class)->constructed (gobj);

    g_assert_nonnull (p->l);
}

static void
set_property (GObject *gobj,
              guint propid,
              const GValue *value,
              GParamSpec *spec)
{
    GannLayer *self = GANN_LAYER (gobj);
    GannLayerPrivate *p = gann_layer_get_instance_private (self);

    switch (propid)
    {
    case PROP_NETWORK:
        g_set_weak_pointer (&p->network, g_value_get_object (value));
        break;

    case PROP_WIDTH:
        p->width = g_value_get_int (value);
        break;

    case PROP_HEIGHT:
        p->height = g_value_get_int (value);
        break;

    case PROP_DEPTH:
        p->depth = g_value_get_int (value);
        break;

    case PROP_ACTIVATION:
        g_clear_pointer (&p->activation, g_free);
        p->activation = g_value_dup_string (value);
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
    GannLayer *self = GANN_LAYER (gobj);
    GannLayerPrivate *p = gann_layer_get_instance_private (self);

    switch (propid)
    {
    case PROP_NETWORK:
        g_value_set_object (value, p->network);
        break;

    case PROP_WIDTH:
        g_value_set_int (value, p->width);
        break;

    case PROP_HEIGHT:
        g_value_set_int (value, p->height);
        break;

    case PROP_DEPTH:
        g_value_set_int (value, p->depth);
        break;

    case PROP_ACTIVATION:
        g_value_set_string (value, p->activation);
        break;

    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (gobj, propid, spec);
    }
}

const gfloat *
gann_layer_get_data (GannLayer *self,
                     gsize *size)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);

    if (p->value_buff == NULL) {
        p->value_buff = g_new (gfloat, p->l->size);
    }

    layer_load_value (p->l, p->value_buff, 0, p->l->size);

    if (size != NULL) {
        *size = p->l->size;
    }

    return p->value_buff;
}

GannNetwork *
gann_layer_get_network (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    return p->network;
}

gint
gann_layer_get_width (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    return p->width;
}

gint
gann_layer_get_height (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    return p->height;
}

gint
gann_layer_get_depth (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    return p->depth;
}

const gchar *
gann_layer_get_activation (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    return p->activation;
}

/* PRIVATE */

void
gann_layer_set_core (GannLayer *self,
                     struct layer *core)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    g_assert (p->l == NULL);
    p->l = core;
}

struct layer *
gann_layer_get_core (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    return p->l;
}

GannLayer *
gann_layer_new_input (GannNetwork *network,
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

GannLayer *
gann_layer_new_output (GannNetwork *network)
{
    return g_object_new (GANN_TYPE_OUTPUT_LAYER,
                         "network", network,
                         NULL);
}

GannLayer *
gann_layer_new_dense (GannNetwork *network,
                      gint width,
                      gint height,
                      gint depth,
                      const gchar *activation)
{
    return g_object_new (GANN_TYPE_DENSE_LAYER,
                         "network", network,
                         "width", width,
                         "height", height,
                         "depth", depth,
                         "activation", activation,
                         NULL);
}
