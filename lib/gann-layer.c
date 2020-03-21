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

#include "gann-input-layer.h"
#include "gann-output-layer.h"
#include "gann-dense-layer.h"
#include "gann-conv-layer.h"
#include "gann-network.h"

#include "core/layer.h"

typedef struct {
    GannNetwork *network;

    gint width;
    gint height;
    gint depth;
    gchar *activation;

    gfloat *value_buff;
    guint8 *bytes_buff;

    GSList *prev_list;
    GSList *next_list;

    struct layer *l;
} GannLayerPrivate;

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (GannLayer, gann_layer, G_TYPE_OBJECT);

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
static void compile (GannLayer *self);

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

    cls->compile = compile;

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
                          -1, G_MAXINT16, 0,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT |
                          G_PARAM_STATIC_STRINGS);
    props[PROP_HEIGHT] =
        g_param_spec_int ("height",
                          "Height",
                          "Layer height",
                          -1, G_MAXINT16, 0,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT |
                          G_PARAM_STATIC_STRINGS);

    props[PROP_DEPTH] =
        g_param_spec_int ("depth",
                          "Depth",
                          "Layer depth",
                          -1, G_MAXINT16, 0,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT |
                          G_PARAM_STATIC_STRINGS);

    props[PROP_ACTIVATION] =
        g_param_spec_string ("activation",
                             "Activation",
                             "Layer activation",
                             "linear",
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
    g_clear_pointer (&p->bytes_buff, g_free);

    G_OBJECT_CLASS (gann_layer_parent_class)->dispose (gobj);
}

static void
constructed (GObject *gobj)
{
    GannLayer *self = GANN_LAYER (gobj);
    GannLayerPrivate *p = gann_layer_get_instance_private (self);

    g_assert_nonnull (p->network);
    gann_network_attach_layer (p->network, self);

    p->value_buff = NULL;
    p->bytes_buff = NULL;
    p->next_list = NULL;
    p->prev_list = NULL;

    G_OBJECT_CLASS (gann_layer_parent_class)->constructed (gobj);
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

static void
compile (GannLayer *self)
{
    g_message ("Gann::Layer::compile");
}

/**
 * gann_layer_compile: (virtual compile)
 *
 * Compiles the layer
 */
void
gann_layer_compile (GannLayer *self)
{
    GANN_LAYER_GET_CLASS (self)->compile (self);
}

/**
 * gann_layer_append:
 * @next: next layer to append
 *
 * Appends new layer in front
 *
 * returns: (transfer none): self
 */
GannLayer *
gann_layer_append (GannLayer *self,
                   GannLayer *next)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);

    if (g_slist_find (p->next_list, next) == NULL) {
        p->next_list = g_slist_prepend (p->next_list, next);
        layer_append (gann_layer_get_core (self),
                    gann_layer_get_core (next));
        gann_layer_prepend (next, self);
    }

    return self;
}

/**
 * gann_layer_prepend
 * @prev: previous layer to prepend
 *
 * Prepends new layer in back
 *
 * returns: (transfer none): self
 */
GannLayer *
gann_layer_prepend (GannLayer *self,
                    GannLayer *prev)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);

    if (g_slist_find (p->prev_list, prev) == NULL) {
        p->prev_list = g_slist_prepend (p->prev_list, prev);
        layer_prepend (gann_layer_get_core (self),
                    gann_layer_get_core (prev));
        gann_layer_append (prev, self);
    }

    return self;
}

/**
 * gann_layer_next_list:
 * 
 * returns: (transfer none) (element-type GannLayer): next layers list
 */
GSList *
gann_layer_next_list (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);

    return p->next_list;
}

/**
 * gann_layer_prev_list:
 *
 * returns: (transfer none) (element-type GannLayer): prev layers list
 */
GSList *
gann_layer_prev_list (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);

    return p->prev_list;
}

/**
 * gann_layer_get_data:
 * @size: return array's size
 *
 * returns: (array length=size) (transfer none): Pointer to layer's data
 */
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

/**
 * gann_layer_get_data_bytes:
 * 
 * returns: (array length=size) (transfer none): pointer to layer's data
 * converted to bytes
 */
const guint8 *
gann_layer_get_data_bytes (GannLayer *self,
                           gsize *size)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    gint i;

    gann_layer_get_data (self, size);

    if (p->bytes_buff == NULL) {
        p->bytes_buff = g_new (guint8, p->l->size);
    }

    g_message ("size: %lu", *size);
    for (i = 0; i < *size; i++) {
        p->bytes_buff[i] = p->value_buff[i] * 255.0f;
    }

    return p->bytes_buff;
}

/**
 * gann_layer_get_network:
 *
 * returns: (transfer none): Pointer to network instance
 */
GannNetwork *
gann_layer_get_network (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    return p->network;
}

/**
 * gann_layer_get_width:
 *
 * returns: network width
 */
gint
gann_layer_get_width (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    return p->width;
}

/**
 * gann_layer_get_height:
 *
 * returns: network height
 */
gint
gann_layer_get_height (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    return p->height;
}

/**
 * gann_layer_get_depth:
 *
 * returns: network depth
 */
gint
gann_layer_get_depth (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    return p->depth;
}

/**
 * gann_layer_get_activation:
 *
 * returns: (transfer none): Name of activation function
 */
const gchar *
gann_layer_get_activation (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    return p->activation;
}

/**
 * gann_layer_set_core: (skip)
 */
void
gann_layer_set_core (GannLayer *self,
                     struct layer *core)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    g_assert (p->l == NULL);
    p->l = core;
}

/**
 * gann_layer_get_core: (skip)
 */
struct layer *
gann_layer_get_core (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    return p->l;
}
