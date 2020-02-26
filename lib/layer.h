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
    int weight_c;
    int value_c;
    int width;
    int height;
    int depth;
    enum activation_type activation;
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

void layer_forward (struct layer *lay);
void layer_backward (struct layer *lay);
void layer_activate (struct layer *lay);
void layer_free (struct layer *lay);
