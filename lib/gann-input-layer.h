/*
 * gann-input-layer.h
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

G_BEGIN_DECLS

#define GANN_TYPE_INPUT_LAYER (gann_input_layer_get_type ())

G_DECLARE_FINAL_TYPE (GannInputLayer, gann_input_layer,
                      GANN, INPUT_LAYER, GannLayer);

/**
 * gann_input_layer_new:
 *
 * returns: (transfer full): New input layer instance
 */
GannInputLayer *gann_input_layer_new (gint width,
                                      gint height,
                                      gint depth);

/**
 * gann_input_layer_set_data:
 * @data: (array length=datasize): float array
 */
void gann_input_layer_set_data (GannInputLayer *self,
                                const gfloat *data,
                                gsize datasize);

/**
 * gann_input_layer_set_data_bytes:
 * @data: (array length=datasize): byte array
 */
void gann_input_layer_set_data_bytes (GannInputLayer *self,
                                      const guint8 *data,
                                      gsize datasize);

G_END_DECLS
