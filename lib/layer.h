#pragma once

#include <glib.h>

enum layer_type
{
    LAYER_NONE,
    LAYER_INPUT,
    LAYER_CONVOLUTION,
    LAYER_ACTIVATION,
    LAYER_FULLY,
    N_LAYERS,
};

enum activation_type
{
    ACTIVATION_NONE,
    ACTIVATION_LINEAR,
    ACTIVATION_RELU,
    ACTIVATION_SIGMOID,
    N_ACTIVATIONS,
};

struct layer
{
    struct network *net;
    enum layer_type type;
    float *weight_v;
    float *value_v;
    float *bias_v;
    int weight_c;
    int value_c;
    int bias_c;
    int width;
    int height;
    int depth;
    void (*forward) (struct layer *lay);
    void (*backward) (struct layer *lay);
    void (*release) (struct layer *lay);
};

struct layer *layer_make_input (struct network *net,
                                int width, int height, int depth);
struct layer *layer_make_convolution (struct network *net,
                                      int range, int filterc);
struct layer *layer_make_activation (struct network *net,
                                     enum activation_type type);
struct layer *layer_make_full (struct network *net,
                               int width, int height, int depth);

void layer_forward (struct layer *lay);
void layer_backward (struct layer *lay);
