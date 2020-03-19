/*
 * gann-layer.h
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

#pragma once

#include <glib-object.h>

G_BEGIN_DECLS

struct layer;
typedef struct _GannNetwork GannNetwork;

#define GANN_TYPE_LAYER (gann_layer_get_type ())

G_DECLARE_DERIVABLE_TYPE (GannLayer, gann_layer,
                          GANN, LAYER, GObject);

struct _GannLayerClass
{
    GObjectClass parent_class;

    /**
     * attached:
     *
     * Called when attached to GannNetwork instance
     */
    void (*attached) (GannLayer *self);
};

/**
 * gann_layer_attach:
 * @network: Network instance to attach. Network takes
 *           Layer ownership after the call.
 *
 * returns: (transfer none): self
 */
GannLayer *gann_layer_attach (GannLayer *self,
                              GannNetwork *network);

/**
 * gann_layer_get_data:
 * @size: return array's size
 *
 * returns: (array length=size) (transfer none): Pointer to layer's data
 */
const gfloat *gann_layer_get_data (GannLayer *self,
                                   gsize *size);

/**
 * gann_layer_get_data_bytes:
 * 
 * returns: (array length=size) (transfer none): pointer to layer's data
 * converted to bytes
 */
const guint8 *gann_layer_get_data_bytes (GannLayer *self,
                                         gsize *size);

/**
 * gann_layer_get_core:
 *
 * returns: (transfer none): Pointer to underlying core structure
 */
struct layer *gann_layer_get_core (GannLayer *self);

/**
 * gann_layer_get_network:
 *
 * returns: (transfer none): Pointer to network instance
 */
GannNetwork *gann_layer_get_network (GannLayer *self);
gint gann_layer_get_width (GannLayer *self);
gint gann_layer_get_height (GannLayer *self);
gint gann_layer_get_depth (GannLayer *self);

/**
 * gann_layer_get_activation:
 *
 * returns: (transfer none): Name of activation function
 */
const gchar *gann_layer_get_activation (GannLayer *self);

G_END_DECLS
