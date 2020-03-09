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
 * along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "gann-layer.h"

G_BEGIN_DECLS

#define GANN_TYPE_INPUT_LAYER (gann_input_layer_get_type ())

G_DECLARE_FINAL_TYPE (GannInputLayer, gann_input_layer,
                      GANN, INPUT_LAYER, GannLayer);

void gann_input_layer_set_input (GannInputLayer *self,
                                 const gfloat *data,
                                 gsize datasize);
void gann_input_layer_set_input_floats (GannInputLayer *self,
                                        gfloat first, ...);

G_END_DECLS
