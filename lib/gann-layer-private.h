/*
 * gann-layer-private.h
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

#include "gann-opencl.h"
#include "gann-layer.h"

G_BEGIN_DECLS

void gann_layer_clear_gradient (GannLayer *self);
void gann_layer_create_buffer (GannLayer *self,
                               cl_mem *handle,
                               cl_int size,
                               cl_int flags);

G_END_DECLS
