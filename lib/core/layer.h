/*
 * layer.h
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

#pragma once

#include "context.h"

enum layer_type
{
    LAYER_NONE,
    LAYER_INPUT,
    LAYER_OUTPUT,
    LAYER_CONVOLUTION,
    LAYER_ACTIVATION,
    LAYER_DENSE,
    N_LAYERS,
};

struct layer
{
    struct network *net;
    struct layer *prev;
    enum layer_type type;
    const char *activation;

    cl_mem value_mem;
    cl_mem derivative_mem;
    cl_mem gradient_mem;
    cl_mem bias_mem;
    cl_mem bias_delta_mem;

    cl_mem weight_mem;
    cl_mem delta_mem;

    cl_program program;
    cl_kernel kernels[16];

    int width;
    int height;
    int depth;
    int size;
    int weights;

    void (*compile) (struct layer *lay);
    void (*forward) (struct layer *lay);
    void (*backward) (struct layer *lay);
    void (*release) (struct layer *lay);
};

struct layer *layer_make_convolution (struct network *net,
                                      const char *activation,
                                      int kernel_width, int kernel_height,
                                      int filter_count);
struct layer *layer_make_full (struct network *net,
                               const char *activation,
                               int width, int height, int depth);
struct layer *layer_make_input (struct network *net,
                                int width, int height, int depth);
struct layer *layer_make_output (struct network *net);

void layer_compile (struct layer *lay);
int layer_is_compiled (struct layer *lay);
void layer_forward (struct layer *lay);
void layer_backward (struct layer *lay);
void layer_free (struct layer *lay);
void layer_load_value (struct layer *lay,
                       float *buff,
                       int offset,
                       int count);
void layer_clear_gradient (struct layer *lay);

void layer_create_buffer (struct layer *lay,
                          cl_mem *handle,
                          int size,
                          int flags);

void layer_input_set_data (struct layer *lay,
                           const float *data,
                           int size);

void layer_output_set_truth (struct layer *lay,
                             const float *data,
                             int size);
