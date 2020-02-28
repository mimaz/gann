#pragma once

#include <glib.h>
#include <math.h>

#define LEAKY_ALPHA (1.0f / 256.0f)

enum activation_type
{
    ACTIVATION_NONE,
    ACTIVATION_LINEAR,
    ACTIVATION_RELU,
    ACTIVATION_SIGMOID,
    ACTIVATION_LEAKY,
    N_ACTIVATIONS,
};

static inline float activation_value (enum activation_type type,
                                      float input)
{
    switch (type) {
    case ACTIVATION_NONE:
    case ACTIVATION_LINEAR:
        return input;

    case ACTIVATION_RELU:
        return MAX (input, 0);

    case ACTIVATION_SIGMOID:
        return 1.0f / (1.0f + expf (-input));

    case ACTIVATION_LEAKY:
        return input > 0 ? input : (input * LEAKY_ALPHA);

    default:
        g_abort ();
    }
}

static inline float activation_derivative (enum activation_type type,
                                           float value)
{
    switch (type) {
    case ACTIVATION_NONE:
    case ACTIVATION_LINEAR:
        return 1;

    case ACTIVATION_RELU:
        return value > 0 ? 1 : 0;

    case ACTIVATION_SIGMOID:
        return value * (1 - value);

    case ACTIVATION_LEAKY:
        return value > 0 ? 1 : LEAKY_ALPHA;

    default:
        g_abort ();
    }
}
