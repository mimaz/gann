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

#include "gann-buffer.h"

G_BEGIN_DECLS

struct layer;
typedef struct _GannNetwork GannNetwork;
typedef struct _GannContext GannContext;
typedef struct _GannBarrier GannBarrier;

#define GANN_TYPE_LAYER (gann_layer_get_type ())

G_DECLARE_DERIVABLE_TYPE (GannLayer, gann_layer,
                          GANN, LAYER, GObject);

struct _GannLayerClass
{
    GObjectClass parent_class;

    /**
     * forward:
     *
     * Propagates layer forward
     */
    void (*forward) (GannLayer *self);

    /**
     * backward:
     *
     * Backpropagates layer
     */
    void (*backward) (GannLayer *self);

    /**
     * compile:
     *
     * Compile the layer
     */
    void (*compile) (GannLayer *self);

    /**
     * value_buffer:
     *
     * Gets value buffer
     *
     * returns: (transfer none):
     */
    GannBuffer *(*value_buffer) (GannLayer *self);

    /**
     * gradient_buffer:
     *
     * Gets gradient buffer
     *
     * returns: (transfer none):
     */
    GannBuffer *(*gradient_buffer) (GannLayer *self);
};

void gann_layer_forward (GannLayer *self);
void gann_layer_backward (GannLayer *self);
void gann_layer_compile (GannLayer *self);
GannLayer *gann_layer_append (GannLayer *self,
                              GannLayer *next);
GannLayer *gann_layer_prepend (GannLayer *self,
                               GannLayer *prev);
GSList *gann_layer_next_list (GannLayer *self);
GSList *gann_layer_prev_list (GannLayer *self);
GannLayer *gann_layer_next_layer (GannLayer *self);
GannLayer *gann_layer_prev_layer (GannLayer *self);
GannBuffer *gann_layer_value_buffer (GannLayer *self);
GannBuffer *gann_layer_gradient_buffer (GannLayer *self);
const gfloat *gann_layer_get_data (GannLayer *self,
                                   gsize *size);
const guint8 *gann_layer_get_data_bytes (GannLayer *self,
                                         gsize *size);
GannNetwork *gann_layer_get_network (GannLayer *self);
GannContext *gann_layer_get_context (GannLayer *self);
gint gann_layer_get_width (GannLayer *self);
gint gann_layer_get_height (GannLayer *self);
gint gann_layer_get_depth (GannLayer *self);
const gchar *gann_layer_get_activation (GannLayer *self);
void gann_layer_set_propagated (GannLayer *self,
                                gboolean propagated);
gboolean gann_layer_get_propagated (GannLayer *self);
void gann_layer_set_compiled (GannLayer *self,
                              gboolean compiled);
gboolean gann_layer_get_compiled (GannLayer *self);
void gann_layer_set_forward_barrier (GannLayer *self,
                                     GannBarrier *barrier);
GannBarrier *gann_layer_get_forward_barrier (GannLayer *self);
void gann_layer_set_backward_barrier (GannLayer *self,
                                      GannBarrier *barrier);
GannBarrier *gann_layer_get_backward_barrier (GannLayer *self);
void gann_layer_set_core (GannLayer *self,
                          struct layer *core);
struct layer *gann_layer_get_core (GannLayer *self);

G_END_DECLS
