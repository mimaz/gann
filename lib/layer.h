#pragma once

#include "activation.h"

enum layer_type
{
    LAYER_NONE,
    LAYER_INPUT,
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
    float *value_v;
    float *gradient_v;
    float *weight_v;
    float *delta_v;
    int weight_c;
    int value_c;
    int width;
    int height;
    int depth;
    enum activation_type activation;
    void (*forward) (struct layer *lay);
    void (*backward) (struct layer *lay);
    void (*release) (struct layer *lay);
    void (*loss) (struct layer *lay);
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

void layer_free (struct layer *lay);
void layer_randomize (struct layer *lay);
