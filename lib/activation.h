#pragma once

#include <glib.h>
#include <math.h>

enum activation_type
{
    ACTIVATION_NONE,
    ACTIVATION_LINEAR,
    ACTIVATION_RELU,
    ACTIVATION_SIGMOID,
    N_ACTIVATIONS,
};

static inline void activation_value (enum activation_type type,
                                     float input,
                                     float *value,
                                     float *derivative)
{
    float tmp;

    switch (type) {
    case ACTIVATION_NONE:
    case ACTIVATION_LINEAR:
        if (value != NULL) {
            *value = input;
        }
        if (derivative != NULL) {
            *derivative = 1;
        }
        break;

    case ACTIVATION_RELU:
        if (value != NULL) {
            *value = MAX (input, 0);
        }
        if (derivative != NULL) {
            *derivative = input > 0 ? 1 : 0;
        }
        break;

    case ACTIVATION_SIGMOID:
        tmp = 1.0f / (1.0f + expf (-input));

        if (value != NULL) {
            *value = tmp;
        }
        if (derivative != NULL) {
            *derivative = tmp * (1 - tmp);
        }
        break;

    default:
        g_abort ();
    }
}
