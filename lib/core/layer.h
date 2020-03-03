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
    LAYER_FULLY,
    N_LAYERS,
};

struct layer
{
    struct network *net;
    struct layer *prev;
    enum layer_type type;
    enum activation_type activation;

    float *value_v;
    cl_mem value_mem;
    float *gradient_v;
    cl_mem gradient_mem;
    float *bias_v;
    cl_mem bias_mem;

    float *weight_v;
    cl_mem weight_mem;
    float *delta_v;
    cl_mem delta_mem;
    float *bias_delta_v;
    cl_mem bias_delta_mem;

    cl_program program;
    cl_kernel kernels[8];

    int width;
    int height;
    int depth;
    int size;
    int weights;

    void (*forward) (struct layer *lay);
    void (*backward) (struct layer *lay);
    void (*release) (struct layer *lay);
    void (*loss) (struct layer *lay);
    void (*initialize) (struct layer *lay);
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

void layer_free (struct layer *lay);
void layer_randomize (struct layer *lay);

void layer_output_set_truth (struct layer *lay,
                             const float *data,
                             int size);
