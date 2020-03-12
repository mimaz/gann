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
};

GannLayer *gann_layer_new_input (GannNetwork *network,
                                 gint width,
                                 gint height,
                                 gint depth);
GannLayer *gann_layer_new_output (GannNetwork *network);
GannLayer *gann_layer_new_dense (GannNetwork *network,
                                 gint width,
                                 gint height,
                                 gint depth,
                                 const gchar *activation);
GannLayer *gann_layer_new_conv (GannNetwork *network,
                                gint size,
                                gint stride,
                                gint filters,
                                const gchar *activation);

const gfloat *gann_layer_get_data (GannLayer *self,
                                   gsize *size);
struct layer *gann_layer_get_core (GannLayer *self);

GannNetwork *gann_layer_get_network (GannLayer *self);
gint gann_layer_get_width (GannLayer *self);
gint gann_layer_get_height (GannLayer *self);
gint gann_layer_get_depth (GannLayer *self);
const gchar *gann_layer_get_activation (GannLayer *self);

G_END_DECLS
