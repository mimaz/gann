/*
 * gann-buffer.h
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

typedef struct _GannContext GannContext;

#define GANN_TYPE_BUFFER (gann_buffer_get_type ())

struct _GannBufferClass
{
    GObjectClass parent_class;
};

G_DECLARE_DERIVABLE_TYPE (GannBuffer, gann_buffer,
                          GANN, BUFFER, GObject);

GannBuffer *gann_buffer_new (GannContext *context,
                             GType element_type,
                             gint width,
                             gint height,
                             gint depth);
GannContext *gann_buffer_get_context (GannBuffer *self);
GType gann_buffer_get_element_type (GannBuffer *self);
gint gann_buffer_get_element_size (GannBuffer *self);
gint gann_buffer_get_width (GannBuffer *self);
gint gann_buffer_get_height (GannBuffer *self);
gint gann_buffer_get_depth (GannBuffer *self);
GannBuffer *gann_buffer_write (GannBuffer *self,
                               gint offset,
                               const gfloat *data,
                               gsize size);
const gfloat *gann_buffer_read (GannBuffer *self,
                                gint offset,
                                gint count,
                                gsize *size);

G_END_DECLS
