/*
 * gann-context.h
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

struct context;
typedef struct _GannNetwork GannNetwork;

#define GANN_TYPE_CONTEXT (gann_context_get_type ())

G_DECLARE_FINAL_TYPE (GannContext, gann_context,
                      GANN, CONTEXT, GObject);

/**
 * gann_context_new:
 *
 * returns: (transfer full): New context instance
 */
GannContext *gann_context_new ();
void gann_context_add_network (GannContext *self,
                               GannNetwork *network);
void gann_context_remove_network (GannContext *self,
                                  GannNetwork *network);
struct context *gann_context_get_core (GannContext *self);

G_END_DECLS
