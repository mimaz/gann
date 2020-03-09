#pragma once

#include "activation.h"
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
    enum activation_type activation;

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
                                      enum activation_type activation,
                                      int kernel_width, int kernel_height,
                                      int filter_count);
struct layer *layer_make_full (struct network *net,
                               enum activation_type activation,
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
