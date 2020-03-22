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
 * along with Gann.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "context.h"

#define LAYER_FLAG_COMPILED 1

enum layer_type
{
    LAYER_NONE,
    LAYER_INPUT,
    LAYER_OUTPUT,
    LAYER_CONV,
    LAYER_DENSE,
    N_LAYERS,
};

struct layer
{
    /*
     * network pointer
     */
    struct network *net;

    /*
     * previous and next layer pointers
     */
    struct layer *prev;
    struct layer *next;

    /*
     * layer type
     */
    enum layer_type type;

    /*
     * state flags
     */
    int flags;

    /*
     * activation name
     */
    const char *activation;

    /*
     * common memory buffers
     * with length equals the number of output nodes
     */
    cl_mem value_mem;
    cl_mem derivative_mem;
    cl_mem gradient_mem;
    cl_mem bias_mem;
    cl_mem bias_delta_mem;

    /*
     * common memory buffers
     * with length equals the number of node connections
     */
    cl_mem weight_mem;
    cl_mem delta_mem;

    /*
     * barrier events
     */
    cl_event forward_barrier;
    cl_event backward_barrier;

    /*
     * 3D size
     */
    int width;
    int height;
    int depth;

    /*
     * 1D size, equal to width * height * depth
     */
    int size;

    /*
     * number of weights
     */
    int weights;

    /*
     * Layer's loss, sum of all layers is the total loss
     */
    float loss;

    /*
     * virtual functions, layer type specific
     */
    void (*compile) (struct layer *lay);
    void (*forward) (struct layer *lay);
    void (*backward) (struct layer *lay);
    void (*release) (struct layer *lay);
};

/*
 * layer_make_dense:
 * Creates dense layer
 * width: width dimension
 * height: height dimension
 * depth: depth dimension
 * activation: activation name, should be valid at least
 * for the call time
 * prev: previous layer pointer or NULL to attach to last one
 */
struct layer *layer_make_dense (struct network *net,
                                int width, int height, int depth,
                                const char *activation,
                                struct layer *prev);

/*
 * layer_make_conv:
 * Creates convolutional layer
 * size: size of the kernel, for example 3 for 3x3 kernel
 * depth: number of filters
 * activation: activation function name
 * prev: previous layer or NULL to attach to the last one
 */
struct layer *layer_make_conv (struct network *net,
                               int size, int depth, int stride,
                               const char *activation,
                               struct layer *prev);

/*
 * layer_make_input:
 * Creates input layer
 * width: width dimension
 * height: height dimension
 * depth: depth dimension
 */
struct layer *layer_make_input (struct network *net,
                                int width, int height, int depth);

/*
 * layer_make_output
 * Creates output layer
 */
struct layer *layer_make_output (struct network *net);

/*
 * layer_append
 * Appends another layer in front
 */
void layer_append (struct layer *lay,
                   struct layer *other);

/*
 * layer_prepend
 * Prepends another layer back
 */
void layer_prepend (struct layer *lay,
                    struct layer *other);

/*
 * layer_compile
 * Compiles the layer if it wasn't compiled yet. Does
 * nothing otherwise
 */
void layer_compile (struct layer *lay);

/*
 * layer_forward:
 * Propagates layer forward
 */
void layer_forward (struct layer *lay);

/*
 * layer_backward:
 * Propagates layer back
 */
void layer_backward (struct layer *lay);

/*
 * layer_free:
 * Frees layer
 */
void layer_free (struct layer *lay);

/*
 * layer_load_value:
 * Loads current layer value into the local buffer
 * buff: buffer of size at least $count allocated by the caller
 * offset: offset of the layer value data
 * count: number of numeric (float) values to read
 */
void layer_load_value (struct layer *lay,
                       float *buff,
                       int offset,
                       int count);

/*
 * layer_clear_gradient:
 * Clears the gradient buffer
 */
void layer_clear_gradient (struct layer *lay);

/*
 * layer_create_buffer:
 * Creates a memory buffer owned by the layer
 * handle: pointer to the buffer handle
 * size: size in number of numeric (float) values
 * flags: OpenCL buffer flags
 */
void layer_create_buffer (struct layer *lay,
                          cl_mem *handle,
                          int size,
                          int flags);

/*
 * layer_input_set_data
 * Sets data for the input layer
 * data: data memory
 * size: number of numeric (float) values to write
 */
void layer_input_set_data (struct layer *lay,
                           const float *data,
                           int size);

/*
 * layer_output_set_truth:
 * Sets truth data to the output layer
 * data: data memory
 * size: number of numeric (float) values
 */
void layer_output_set_truth (struct layer *lay,
                             const float *data,
                             int size);

/*
 * layer_conv_set_filter:
 * Sets convolutional layer filter data
 * data: data memory
 * offset: offset from the beginning of memory
 * size: size of data array, must be equal to actual filter size
 */
void layer_conv_set_filter (struct layer *lay,
                            const float *data,
                            int offset,
                            int size);

/*
 * layer_conv_get_filter:
 * Returns convolutional layer filter data
 * offset: offset from the beginning of memory
 * size: (optional): pointer to returned array size
 */
const float *layer_conv_get_filter (struct layer *lay,
                                    int offset,
                                    int *size);
