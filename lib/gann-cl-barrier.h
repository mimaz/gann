/*
 * gann-cl-barrier.h
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
#include "gann-opencl.h"

G_BEGIN_DECLS

#define GANN_TYPE_CL_BARRIER (gann_cl_barrier_get_type ())

G_DECLARE_INTERFACE (GannClBarrier, gann_cl_barrier,
                     GANN, CL_BARRIER, GObject);

struct _GannClBarrierInterface
{
    GTypeInterface parent_interface;

    gboolean (*forward_barrier) (GannClBarrier *self,
                                 cl_event *event);
    gboolean (*backward_barrier) (GannClBarrier *self,
                                  cl_event *event);
};

G_END_DECLS
