#include "layer.h"
#include "network.h"

#include <math.h>

void
layer_forward (struct layer *lay)
{
    g_message ("forward");
    for (int i = 0; i < lay->net->input_c; i++)
        g_message ("input %d: %f", i, lay->net->input_v[i]);
    for (int i = 0; i < lay->weight_c; i++)
        g_message ("weights %d: %f", i, lay->weight_v[i]);
    lay->forward (lay);
}

void
layer_backward (struct layer *lay)
{
    lay->backward (lay);
}

void
layer_activate (struct layer *lay)
{
    float in, out;
    int i;

    for (i = 0; i < lay->value_c; i++) {
        in = lay->value_v[i];

        switch (lay->activation) {
        case ACTIVATION_NONE:
        case ACTIVATION_LINEAR:
            out = in;
            break;

        case ACTIVATION_RELU:
            out = MAX (0, in);
            break;

        case ACTIVATION_SIGMOID:
            out = 1.0f / (1.0f + expf (-in));
            break;

        default:
            g_abort ();
            break;
        }

        lay->value_v[i] = out;
    }
}

void
layer_free (struct layer *lay)
{
    if (lay->release != NULL) {
        lay->release (lay);
    }

    g_clear_pointer (&lay->weight_v, g_free);
    g_clear_pointer (&lay->value_v, g_free);

    g_free (lay);
}

void
layer_randomize (struct layer *lay)
{
    int i;

    for (i = 0; i < lay->weight_c; i++) {
        lay->weight_v[i] = (rand () % 1000) / 999.0f;
    }
}
