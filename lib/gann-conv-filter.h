/*
 * gann-conv-filter.h
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

#define GANN_TYPE_CONV_FILTER (gann_conv_filter_get_type ())

G_DECLARE_FINAL_TYPE (GannConvFilter, gann_conv_filter,
                      GANN, CONV_FILTER, GObject);

/**
 * gann_conv_filter_new:
 * @width: Filter's width
 * @height: Filter's height
 * @depth: Filter's depth
 *
 * Creates new convolutional filter
 */
GannConvFilter *gann_conv_filter_new (gint width,
                                      gint height,
                                      gint depth);

/**
 * gann_conv_filter_write_row:
 * @layer: layer index (depth)
 * @row: row index (y coordinate)
 * @value_v: (array length=value_c): values array
 *
 * Writes filter row at given position
 *
 * returns: (transfer none): self
 */
GannConvFilter *gann_conv_filter_write_row (GannConvFilter *self,
                                            gint layer,
                                            gint row,
                                            const gfloat *value_v,
                                            gsize value_c);

/**
 * gann_conv_filter_write_column:
 * @layer: layer index (depth)
 * @column: column index (x coordinate)
 * @value_v: (array length=value_c): values array
 *
 * Writer filter column at given position
 *
 * returns: (transfer none): self
 */
GannConvFilter *gann_conv_filter_write_column (GannConvFilter *self,
                                               gint layer,
                                               gint column,
                                               const gfloat *value_v,
                                               gsize value_c);

/**
 * gann_conv_filter_write_value:
 * @layer: layer index
 * @column: column index
 * @row: row index
 * @value: value to write
 *
 * Writes single value to filter
 *
 * returns: (transfer none): self
 */
GannConvFilter *gann_conv_filter_write_value (GannConvFilter *self,
                                              gint layer,
                                              gint column,
                                              gint row,
                                              gfloat value);
gint gann_conv_filter_get_width (GannConvFilter *self);
gint gann_conv_filter_get_height (GannConvFilter *self);
gint gann_conv_filter_get_depth (GannConvFilter *self);
gint gann_conv_filter_get_size (GannConvFilter *self);

G_END_DECLS
