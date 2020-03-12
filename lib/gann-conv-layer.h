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

G_BEGIN_DECLS

#define GANN_TYPE_CONV_LAYER (gann_conv_layer_get_type ())

G_DECLARE_FINAL_TYPE (GannConvLayer, gann_conv_layer,
                      GANN, CONV_LAYER, GannLayer);

gint gann_conv_layer_get_size (GannConvLayer *self);
gint gann_conv_layer_get_stride (GannConvLayer *self);

G_END_DECLS
