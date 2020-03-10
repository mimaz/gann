/*
 * layer.c
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

#include "layer.h"
#include "network.h"

#include <math.h>

void
layer_compile (struct layer *lay)
{
    if ((lay->flags & LAYER_FLAG_COMPILED) == 0) {
        lay->compile (lay);
        g_assert (lay->flags & LAYER_FLAG_COMPILED);
    }
}

void
layer_forward (struct layer *lay)
{
    layer_compile (lay);

    if (lay->forward != NULL) {
        lay->forward (lay);
    }
}

void
layer_backward (struct layer *lay)
{
    layer_compile (lay);

    if (lay->backward != NULL) {
        lay->backward (lay);
    }
}

void
layer_free (struct layer *lay)
{
    if (lay->release != NULL) {
        lay->release (lay);
    }

    g_free (lay);
}

void
layer_load_value (struct layer *lay,
                  float *buff,
                  int offset,
                  int count)
{
    g_assert (lay->value_mem != 0);
    g_assert (offset + count <= lay->size);
    clEnqueueReadBuffer (lay->net->ctx->queue,
                         lay->value_mem,
                         CL_TRUE,
                         offset * sizeof (cl_float),
                         count * sizeof (cl_float),
                         buff, 0, NULL, NULL);
}

void
layer_clear_gradient (struct layer *lay)
{
    cl_float zero;

    if (lay->gradient_mem != 0) {
        zero = 0;
        context_fill_pattern (lay->net->ctx,
                              lay->gradient_mem,
                              0, lay->size * sizeof (cl_float),
                              &zero,
                              sizeof (cl_float),
                              0, NULL, NULL);
    }
}

void
layer_create_buffer (struct layer *lay,
                     cl_mem *handle,
                     int size,
                     int flags)
{
    cl_int err;
    cl_mem mem;

    mem = clCreateBuffer (lay->net->ctx->context,
                          flags,
                          size * sizeof (cl_float),
                          NULL, &err);

    g_assert (err == CL_SUCCESS);

    *handle = mem;
}
