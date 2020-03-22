/*
 * gann-program-builder.h
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

#include <glib-object.h>

G_BEGIN_DECLS

typedef struct _GannContext GannContext;

#define GANN_TYPE_PROGRAM_BUILDER (gann_program_builder_get_type ())

G_DECLARE_FINAL_TYPE (GannProgramBuilder, gann_program_builder,
                      GANN, PROGRAM_BUILDER, GObject);

GannProgramBuilder *gann_program_builder_new (GannContext *context);
void gann_program_builder_clear (GannProgramBuilder *self);
void gann_program_builder_option (GannProgramBuilder *self,
                                  const gchar *fmt,
                                  ...);
void gann_program_builder_activation (GannProgramBuilder *self,
                                      gint index,
                                      const gchar *name);
void gann_program_builder_file (GannProgramBuilder *self,
                                const gchar *name);
void gann_program_builder_code (GannProgramBuilder *self,
                                const gchar *code);
void gann_program_builder_program (GannProgramBuilder *self,
                                   cl_program *handle);
void gann_program_builder_kernel (GannProgramBuilder *self,
                                  const gchar *name,
                                  cl_kernel *handle);
void gann_program_builder_build (GannProgramBuilder *self);

G_END_DECLS
