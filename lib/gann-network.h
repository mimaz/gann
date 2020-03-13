/*
 * gann-network.h
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

struct network;
typedef struct _GannContext GannContext;
typedef struct _GannLayer GannLayer;
typedef struct _GannInputLayer GannInputLayer;
typedef struct _GannOutputLayer GannOutputLayer;
typedef struct _GannDenseLayer GannDenseLayer;
typedef struct _GannConvLayer GannConvLayer;

#define GANN_TYPE_NETWORK (gann_network_get_type ())

G_DECLARE_DERIVABLE_TYPE (GannNetwork, gann_network,
                          GANN, NETWORK, GObject);

struct _GannNetworkClass
{
    GObjectClass parent_class;
};

/**
 * gann_network_new:
 *
 * returns: (transfer full): New network instance
 */
GannNetwork *gann_network_new (GannContext *context,
                               gfloat rate,
                               gfloat momentum,
                               gfloat decay);

/**
 * gann_network_create_input:
 *
 * returns: (transfer full): New input layer instance
 */
GannInputLayer *gann_network_create_input (GannNetwork *self,
                                           gint width,
                                           gint height,
                                           gint depth);

/**
 * gann_network_create_output:
 *
 * returns: (transfer full): New output layer instance
 */
GannOutputLayer *gann_network_create_output (GannNetwork *self);

/**
 * gann_network_create_dense:
 *
 * returns: (transfer full): New dense layer instance
 */
GannDenseLayer *gann_network_create_dense (GannNetwork *self,
                                           gint width,
                                           gint height,
                                           gint depth,
                                           const gchar *activation);

/**
 * gann_network_create_conv:
 *
 * returns: (transfer full): New conv layer instance
 */
GannConvLayer *gann_network_create_conv (GannNetwork *self,
                                         gint size,
                                         gint stride,
                                         gint filters,
                                         const gchar *activation);

void gann_network_forward (GannNetwork *self);
void gann_network_backward (GannNetwork *self);

/**
 * gann_network_get_layer:
 *
 * returns: (transfer none): Pointer to layer at index @index
 */
GannLayer *gann_network_get_layer (GannNetwork *self,
                                   gint index);

/**
 * gann_network_get_core:
 *
 * returns: (transfer none): Pointer to underlying core structure
 */
struct network *gann_network_get_core (GannNetwork *self);

/**
 * gann_network_get_context:
 *
 * returns: (transfer none): Pointer to context instance
 */
GannContext *gann_network_get_context (GannNetwork *self);
void gann_network_set_rate (GannNetwork *self,
                            gfloat rate);
gfloat gann_network_get_rate (GannNetwork *self);
void gann_network_set_momentum (GannNetwork *self,
                                gfloat momentum);
gfloat gann_network_get_momentum (GannNetwork *self);
void gann_network_set_decay (GannNetwork *self,
                             gfloat decay);
gfloat gann_network_get_decay (GannNetwork *self);
gint gann_network_layer_count (GannNetwork *self);
void gann_network_set_loss (GannNetwork *self,
                            gfloat loss);
gfloat gann_network_get_loss (GannNetwork *self);
void gann_network_set_average_loss (GannNetwork *self,
                                    gfloat loss);
gfloat gann_network_get_average_loss (GannNetwork *self);

G_END_DECLS
