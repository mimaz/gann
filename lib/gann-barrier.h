/*
 * gann-barrier.h
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

#define GANN_TYPE_BARRIER (gann_barrier_get_type ())

G_DECLARE_FINAL_TYPE (GannBarrier, gann_barrier,
                      GANN, BARRIER, GObject);

GannBarrier *gann_barrier_new (GannContext *context);
GannBarrier *gann_barrier_new_from_barrier (GannBarrier *barrier);
GannContext *gann_barrier_get_context (GannBarrier *self);
void gann_barrier_attach (GannBarrier *self,
                          GannBarrier *barrier);

G_END_DECLS
