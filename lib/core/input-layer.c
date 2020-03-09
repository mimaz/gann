/*
 * input-layer.c
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

#include "layer.h"
#include "network.h"
#include "context.h"

struct layer *
layer_make_input (struct network *net,
                  int width, int height, int depth)
{
    struct layer *base;
    int size;

    base = g_new0 (struct layer, 1);
    size = width * height * depth;

    base->net = net;
    base->type = LAYER_INPUT;
    base->width = width;
    base->height = height;
    base->depth = depth;
    base->size = size;
    base->weights = 0;

    layer_create_buffer (base, &base->value_mem,
                         size, CL_MEM_READ_WRITE);
    layer_create_buffer (base, &base->gradient_mem,
                         size, CL_MEM_READ_WRITE);

    network_push_layer (net, base);

    return base;
}

void
layer_input_set_data (struct layer *lay,
                      const float *data,
                      int size)
{
    g_assert (lay->type == LAYER_INPUT);
    g_assert (size == lay->size);

    g_autofree float *buff = g_memdup (data, sizeof (float) * size);

    for (int i = 0; i < size; i++)
        buff[i] = data[i] - 0.5f;

    clEnqueueWriteBuffer (lay->net->ctx->queue,
                          lay->value_mem,
                          CL_TRUE,
                          0, size * sizeof (cl_float),
                          buff,
                          0, NULL, NULL);
    clFinish (lay->net->ctx->queue);
}
