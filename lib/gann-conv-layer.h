/*
 * gann-conv-layer.h
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

#include "gann-layer.h"
#include "gann-conv-filter.h"

G_BEGIN_DECLS

#define GANN_TYPE_CONV_LAYER (gann_conv_layer_get_type ())

G_DECLARE_FINAL_TYPE (GannConvLayer, gann_conv_layer,
                      GANN, CONV_LAYER, GannLayer);

/**
 * gann_conv_layer_new:
 * @kernel_size: size of kernel, N for NxN kernel
 * @stride: kernel stride (usually 1)
 * @filters: number of output filters
 * @activation: activation function name
 */
GannConvLayer *gann_conv_layer_new (gint kernel_size,
                                    gint stride,
                                    gint filters,
                                    const gchar *activation);

/**
 * gann_conv_layer_set_filter:
 * @index: filter index
 * @filter: new filter instance
 *
 * Sets new filter for the layer
 */
void gann_conv_layer_set_filter (GannConvLayer *self,
                                 gint index,
                                 GannConvFilter *filter);

/**
 * gann_conv_layer_get_filter:
 * @index: filter index
 * 
 * returns: (transfer none): filter instance
 */
GannConvFilter *gann_conv_layer_get_filter (GannConvLayer *self,
                                            gint index);

gint gann_conv_layer_get_kernel_size (GannConvLayer *self);
gint gann_conv_layer_get_stride (GannConvLayer *self);
gint gann_conv_layer_get_filters (GannConvLayer *self);

G_END_DECLS
