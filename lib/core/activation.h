#pragma once

#include <glib.h>
#include <math.h>

#define LEAKY_ALPHA 0.01f

enum activation_type
{
    ACTIVATION_LINEAR,
    ACTIVATION_RELU,
    ACTIVATION_SIGMOID,
    ACTIVATION_LEAKY,
    ACTIVATION_ELU,
    ACTIVATION_STEP,
    N_ACTIVATIONS,
};

static inline float activation_value (enum activation_type type,
                                      float input)
{
    switch (type) {
    case ACTIVATION_LINEAR:
        return input;

    case ACTIVATION_RELU:
        return MAX (input, 0);

    case ACTIVATION_SIGMOID:
        return 1.0f / (1.0f + expf (-input));

    case ACTIVATION_LEAKY:
        return input >= 0 ? input : (input * LEAKY_ALPHA);

    case ACTIVATION_ELU:
        return input >= 0 ? input : (expf (input) - 1);

    case ACTIVATION_STEP:
        return input > 0 ? 1 : 0;

    default:
        g_abort ();
    }
}

static inline float activation_derivative (enum activation_type type,
                                           float value)
{
    switch (type) {
    case ACTIVATION_LINEAR:
        return 1;

    case ACTIVATION_RELU:
        return value > 0 ? 1 : 0;

    case ACTIVATION_SIGMOID:
        return value * (1 - value);

    case ACTIVATION_LEAKY:
        return value >= 0 ? 1 : LEAKY_ALPHA;

    case ACTIVATION_ELU:
        return value >= 0 ? 1 : (value + 1);

    case ACTIVATION_STEP:
        return 0;

    default:
        g_abort ();
    }
}
