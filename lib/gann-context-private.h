/*
 * gann-context-private.h
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
#include "gann-context.h"

G_BEGIN_DECLS

cl_command_queue gann_context_cl_queue (GannContext *self);
cl_context gann_context_cl_context (GannContext *self);
cl_device_id gann_context_cl_device (GannContext *self);
const gchar *gann_context_code (GannContext *self,
                                const gchar *filename);
const gchar *gann_context_activation (GannContext *self,
                                      const gchar *name);

G_END_DECLS
