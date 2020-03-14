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
 * along with Gann.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "layer.h"
#include "network.h"
#include "context.h"
#include "util.h"

struct input_layer
{
    struct layer base;
    float *data;
};

static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void compile (struct layer *lay);
static void release (struct layer *lay);

struct layer *
layer_make_input (struct network *net,
                  int width, int height, int depth)
{
    struct input_layer *input;
    struct layer *base;

    input = g_new0 (struct input_layer, 1);
    base = (struct layer *) input;

    base->net = net;
    base->type = LAYER_INPUT;
    base->width = width;
    base->height = height;
    base->depth = depth;
    base->size = width * height * depth;
    base->weights = 0;
    base->forward = forward;
    base->backward = backward;
    base->compile = compile;
    base->release = release;

    input->data = g_new (float, base->size);

    network_push_layer (net, base);

    return base;
}

void
layer_input_set_data (struct layer *lay,
                      const float *data,
                      int size)
{
    struct input_layer *input;

    g_assert (lay->type == LAYER_INPUT);
    g_assert (size == lay->size);

    input = (struct input_layer *) lay;

    memcpy (input->data, data, size * sizeof (float));
}

static void
forward (struct layer *lay)
{
    struct input_layer *input;

    g_assert (lay->type == LAYER_INPUT);

    input = (struct input_layer *) lay;

    clEnqueueWriteBuffer (lay->net->ctx->queue,
                          lay->value_mem,
                          CL_FALSE,
                          0, lay->size * sizeof (cl_float),
                          input->data,
                          0, NULL,
                          &lay->forward_barrier);
}

static void
backward (struct layer *lay)
{
}

static void
compile (struct layer *lay)
{
    layer_create_buffer (lay, &lay->value_mem,
                         lay->size, CL_MEM_READ_WRITE);
    layer_create_buffer (lay, &lay->gradient_mem,
                         lay->size, CL_MEM_READ_WRITE);

    lay->flags |= LAYER_FLAG_COMPILED;
}

static void
release (struct layer *lay)
{
    struct input_layer *input;

    g_assert (lay->type == LAYER_INPUT);

    input = (struct input_layer *) lay;

    g_clear_pointer (&lay->forward_barrier, clReleaseEvent);
    g_clear_pointer (&input->data, g_free);

    clReleaseMemObject (lay->value_mem);
    clReleaseMemObject (lay->gradient_mem);
}
