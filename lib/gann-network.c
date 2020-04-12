/*
 * gann-network.c
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

#include "gann-network.h"

#include "gann-input-layer.h"
#include "gann-output-layer.h"
#include "gann-dense-layer.h"
#include "gann-conv-layer.h"
#include "gann-context.h"

#include "core/core.h"

typedef struct {
    struct network *net;
    GannContext *context;
    GPtrArray *layer_arr;
    GSList *output_list;
    GSList *propagation_list;
    gfloat avg_loss;
    gboolean compiled;
} GannNetworkPrivate;

G_DEFINE_TYPE_WITH_PRIVATE (GannNetwork, gann_network, G_TYPE_OBJECT);

enum
{
    PROP_0,
    PROP_CONTEXT,
    PROP_RATE,
    PROP_MOMENTUM,
    PROP_DECAY,
    PROP_LAYER_COUNT,
    PROP_LOSS,
    PROP_AVERAGE_LOSS,
    PROP_COMPILED,
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
gann_network_init (GannNetwork *self)
{
}

static void
gann_network_class_init (GannNetworkClass *cls)
{
    GObjectClass *gcls = G_OBJECT_CLASS (cls);

    gcls->dispose = dispose;
    gcls->constructed = constructed;
    gcls->set_property = set_property;
    gcls->get_property = get_property;

    props[PROP_CONTEXT] =
        g_param_spec_object ("context",
                             "Context",
                             "Network's context",
                             GANN_TYPE_CONTEXT,
                             G_PARAM_READWRITE |
                             G_PARAM_CONSTRUCT_ONLY |
                             G_PARAM_STATIC_STRINGS);

    props[PROP_RATE] =
        g_param_spec_float ("rate",
                            "Rate",
                            "Learning rate",
                            0.0f, 1.0f, 0.001f,
                            G_PARAM_READWRITE |
                            G_PARAM_STATIC_STRINGS);

    props[PROP_MOMENTUM] =
        g_param_spec_float ("momentum",
                            "Momentum",
                            "Learning momentum",
                            0.0f, 1.0f, 0.9f,
                            G_PARAM_READWRITE |
                            G_PARAM_STATIC_STRINGS);

    props[PROP_DECAY] =
        g_param_spec_float ("decay",
                            "Decay",
                            "Learning decay",
                            0.0f, 1.0f, 1.0f,
                            G_PARAM_READWRITE |
                            G_PARAM_STATIC_STRINGS);

    props[PROP_LAYER_COUNT] =
        g_param_spec_int ("layer-count",
                          "Layer count",
                          "Number of layers",
                          0, G_MAXINT32, 0,
                          G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS);

    props[PROP_LOSS] =
        g_param_spec_float ("loss",
                            "Loss",
                            "Latest loss",
                            0, G_MAXFLOAT, 0,
                            G_PARAM_READWRITE |
                            G_PARAM_STATIC_STRINGS);

    props[PROP_AVERAGE_LOSS] =
        g_param_spec_float ("average-loss",
                            "Average loss",
                            "Average loss",
                            0, G_MAXFLOAT, 0,
                            G_PARAM_READWRITE |
                            G_PARAM_STATIC_STRINGS);

    props[PROP_COMPILED] =
        g_param_spec_boolean ("compiled",
                              "Compiled",
                              "Compiled",
                              FALSE,
                              G_PARAM_READABLE |
                              G_PARAM_STATIC_STRINGS);

    g_object_class_install_properties (gcls, N_PROPS, props);
}

static void
dispose (GObject *gobj)
{
    GannNetwork *self = GANN_NETWORK (gobj);
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    g_clear_pointer (&p->layer_arr, g_ptr_array_unref);
    g_clear_pointer (&p->output_list, g_slist_free);
    g_clear_pointer (&p->propagation_list, g_slist_free);
    g_clear_pointer (&p->net, network_free);
    g_clear_object (&p->context);

    G_OBJECT_CLASS (gann_network_parent_class)->dispose (gobj);
}

static void
constructed (GObject *gobj)
{
    GannNetwork *self = GANN_NETWORK (gobj);
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    p->net = network_create (gann_context_get_core (p->context));
    p->layer_arr = g_ptr_array_new_with_free_func (g_object_unref);
    p->output_list = NULL;
    p->propagation_list = NULL;
    p->avg_loss = -1;

    G_OBJECT_CLASS (gann_network_parent_class)->constructed (gobj);
}

static void
set_property (GObject *gobj,
              guint propid,
              const GValue *value,
              GParamSpec *spec)
{
    GannNetwork *self = GANN_NETWORK (gobj);
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    switch (propid)
    {
    case PROP_CONTEXT:
        g_set_object (&p->context, g_value_get_object (value));
        break;

    case PROP_RATE:
        gann_network_set_rate (self, g_value_get_float (value));
        break;

    case PROP_MOMENTUM:
        gann_network_set_momentum (self, g_value_get_float (value));
        break;

    case PROP_DECAY:
        gann_network_set_decay (self, g_value_get_float (value));
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
    GannNetwork *self = GANN_NETWORK (gobj);
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    switch (propid)
    {
    case PROP_CONTEXT:
        g_value_set_object (value, p->context);
        break;

    case PROP_RATE:
        g_value_set_float (value, p->net->rate);
        break;

    case PROP_MOMENTUM:
        g_value_set_float (value, p->net->momentum);
        break;

    case PROP_DECAY:
        g_value_set_float (value, p->net->decay);
        break;

    case PROP_LAYER_COUNT:
        g_value_set_int (value, p->layer_arr->len);
        break;

    case PROP_LOSS:
        g_value_set_float (value, p->net->loss);
        break;

    case PROP_AVERAGE_LOSS:
        g_value_set_float (value, p->avg_loss);
        break;

    case PROP_COMPILED:
        g_value_set_boolean (value, p->compiled);
        break;

    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (gobj, propid, spec);
    }
}

/**
 * gann_network_new:
 *
 * returns: (transfer full): New network instance
 */
GannNetwork *
gann_network_new (GannContext *context,
                  gfloat rate, gfloat momentum, gfloat decay)
{
    return g_object_new (GANN_TYPE_NETWORK,
                         "context", context,
                         "rate", rate,
                         "momentum", momentum,
                         "decay", decay,
                         NULL);
}

static void
connect_two_last (GannNetwork *self)
{
    GannLayer *first, *second;

    g_message ("lc %d", gann_network_layer_count (self));

    if (gann_network_layer_count (self) > 1) {
        first = gann_network_layer (self, -2);
        second = gann_network_layer (self, -1);

        g_message ("connect %p %p", first, second);
        gann_layer_append (first, second);
    }
}

/**
 * gann_network_create_input:
 *
 * returns: (transfer none): New input layer instance
 */
GannInputLayer *
gann_network_create_input (GannNetwork *self,
                           gint width,
                           gint height,
                           gint depth)
{
    GannInputLayer *input;

    input = gann_input_layer_new (self, width, height, depth);
    connect_two_last (self);

    return input;
}

/**
 * gann_network_create_output:
 *
 * returns: (transfer none): New output layer instance
 */
GannOutputLayer *
gann_network_create_output (GannNetwork *self)
{
    GannOutputLayer *output;

    output = gann_output_layer_new (self);
    connect_two_last (self);

    return output;
}

/**
 * gann_network_create_dense:
 *
 * returns: (transfer none): New dense layer instance
 */
GannDenseLayer *
gann_network_create_dense (GannNetwork *self,
                           gint width,
                           gint height,
                           gint depth,
                           const gchar *activation)
{
    GannDenseLayer *dense;

    dense = gann_dense_layer_new (self, width, height, depth, activation);
    connect_two_last (self);

    return dense;
}

/**
 * gann_network_create_conv:
 *
 * returns: (transfer none): New conv layer instance
 */
GannConvLayer *
gann_network_create_conv (GannNetwork *self,
                          gint size,
                          gint stride,
                          gint filters,
                          const gchar *activation)
{
    GannConvLayer *conv;

    conv = gann_conv_layer_new (self, size, stride, filters, activation);
    connect_two_last (self);

    return conv;
}

/**
 * gann_network_forward:
 *
 * Propagates network forward
 */
void
gann_network_forward (GannNetwork *self)
{
	return;
    GannNetworkPrivate *p;
    GSList *list;

    gann_network_compile (self);
    
    p = gann_network_get_instance_private (self);
    list = p->propagation_list;

    /* gann_network_clear_propagated (self); */

    g_message ("forward list %p", list);
    while (list != NULL) {
        gann_layer_forward (list->data);
        list = list->next;
    }

    /* network_forward (p->net); */
}

/**
 * gann_network_backward:
 *
 * Backpropagates network
 */
void
gann_network_backward (GannNetwork *self)
{
	return;
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    gann_network_compile (self);

    network_backward (p->net);

    if (p->avg_loss < 0) {
        p->avg_loss = p->net->loss;
    } else {
        p->avg_loss *= 0.99f;
        p->avg_loss += p->net->loss * 0.01f;
    }

    g_object_notify_by_pspec (G_OBJECT (self),
                              props[PROP_LOSS]);
    g_object_notify_by_pspec (G_OBJECT (self),
                              props[PROP_AVERAGE_LOSS]);
}

static GSList *
build_propagation_list (GSList *roots)
{
    g_autoptr (GQueue) queue;
    g_autoptr (GHashTable) table;
    GannLayer *layer;
    GSList *list = NULL;

    queue = g_queue_new ();
    table = g_hash_table_new (g_direct_hash, g_direct_equal);

    while (roots != NULL) {
        g_queue_push_head (queue, roots->data);
        g_hash_table_add (table, roots->data);
        roots = roots->next;
    }

    while (!g_queue_is_empty (queue)) {
        layer = g_queue_pop_tail (queue);
        list = g_slist_prepend (list, layer);
        roots = gann_layer_prev_list (layer);

        while (roots != NULL) {
            if (g_hash_table_lookup (table, roots->data) == NULL) {
                g_queue_push_head (queue, roots->data);
                g_hash_table_add (table, roots->data);
            }
            roots = roots->next;
        }
    }

    g_message ("slist len %d", g_slist_length (list));
    return list;
}

/**
 * gann_network_compile:
 *
 * Compiles network
 */
void
gann_network_compile (GannNetwork *self)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    if (p->compiled) {
        return;
    }

    g_ptr_array_foreach (p->layer_arr,
                         (GFunc) gann_layer_compile,
                         NULL);

    p->propagation_list = build_propagation_list (p->output_list);
    p->compiled = TRUE;

    g_object_notify_by_pspec (G_OBJECT (self),
                              props[PROP_COMPILED]);
}

static void
clear_propagated (gpointer layer,
                  gpointer user_data G_GNUC_UNUSED)
{
    gann_layer_set_propagated (layer, FALSE);
}

/**
 * gann_network_clear_propagated:
 *
 * Clears all layers propagated flag property
 */
void
gann_network_clear_propagated (GannNetwork *self)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    g_ptr_array_foreach (p->layer_arr, clear_propagated, NULL);
}

/**
 * gann_network_attach_layer:
 * @layer: layer to attach
 *
 * Adds layer to network, called by GannLayer constructor
 */
void
gann_network_attach_layer (GannNetwork *self,
                           GannLayer *layer)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    g_message ("attach layer %p", layer);
    g_assert (g_ptr_array_find (p->layer_arr, layer, NULL) == FALSE);
    g_ptr_array_insert (p->layer_arr, -1, layer);

    if (GANN_IS_OUTPUT_LAYER (layer)) {
        p->output_list = g_slist_prepend (p->output_list, layer);
    }
}

/**
 * gann_network_layer:
 *
 * returns: (transfer none): Pointer to layer at index @index
 */
GannLayer *
gann_network_layer (GannNetwork *self,
                    gint index)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    if (index < 0) {
        index = gann_network_layer_count (self) + index;
        g_assert (index >= 0);

        return gann_network_layer (self, index);
    }

    g_message ("index %d", index);
    g_return_val_if_fail (index < p->layer_arr->len, NULL);
    return g_ptr_array_index (p->layer_arr, index);
}

/**
 * gann_network_last_layer:
 *
 * returns: (transfer none): Lastly added layer
 */
GannLayer *
gann_network_last_layer (GannNetwork *self)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    if (p->layer_arr->len > 0)
        return p->layer_arr->pdata[p->layer_arr->len - 1];

    return NULL;
}

/**
 * gann_network_get_core:
 *
 * returns: (transfer none): Pointer to underlying core structure
 */
struct network *
gann_network_get_core (GannNetwork *self)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    return p->net;
}

/**
 * gann_network_get_context:
 *
 * returns: (transfer none): Pointer to context instance
 */
GannContext *
gann_network_get_context (GannNetwork *self)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    return p->context;
}

void
gann_network_set_rate (GannNetwork *self,
                       gfloat rate)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    if (rate != p->net->rate) {
        p->net->rate = rate;
        g_object_notify_by_pspec (G_OBJECT (self),
                                  props[PROP_RATE]);
    }
}

gfloat
gann_network_get_rate (GannNetwork *self)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    return p->net->rate;
}

void
gann_network_set_momentum (GannNetwork *self,
                           gfloat momentum)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    if (momentum != p->net->momentum) {
        p->net->momentum = momentum;
        g_object_notify_by_pspec (G_OBJECT (self),
                                  props[PROP_MOMENTUM]);
    }
}

gfloat
gann_network_get_momentum (GannNetwork *self)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    return p->net->momentum;
}

void
gann_network_set_decay (GannNetwork *self,
                        gfloat decay)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    if (decay != p->net->decay) {
        p->net->decay = decay;
        g_object_notify_by_pspec (G_OBJECT (self),
                                  props[PROP_DECAY]);
    }
}

gfloat
gann_network_get_decay (GannNetwork *self)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    return p->net->decay;
}

gint
gann_network_layer_count (GannNetwork *self)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    return p->layer_arr->len;
}

void
gann_network_set_loss (GannNetwork *self,
                       gfloat loss)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    if (loss != p->net->loss) {
        p->net->loss = loss;
        g_object_notify_by_pspec (G_OBJECT (self),
                                  props[PROP_LOSS]);
    }
}

gfloat
gann_network_get_loss (GannNetwork *self)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    return p->net->loss;
}

void
gann_network_set_average_loss (GannNetwork *self,
                               gfloat loss)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    if (loss != p->net->loss) {
        p->avg_loss = loss;
        g_object_notify_by_pspec (G_OBJECT (self),
                                  props[PROP_LOSS]);
    }
}

gfloat
gann_network_get_average_loss (GannNetwork *self)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    return p->avg_loss;
}
