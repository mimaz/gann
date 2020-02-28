#include "layer.h"
#include "network.h"

#include <math.h>

void
layer_free (struct layer *lay)
{
    if (lay->release != NULL) {
        lay->release (lay);
    }

    g_clear_pointer (&lay->value_v, g_free);
    g_clear_pointer (&lay->delta_v, g_free);
    g_clear_pointer (&lay->gradient_v, g_free);

    g_free (lay);
}

void
layer_randomize (struct layer *lay)
{
    int i;

    for (i = 0; i < lay->weight_c; i++) {
        lay->delta_v[i] = ((float) rand () / RAND_MAX - 0.5f) / lay->weight_c;
        lay->weight_v[i] = 0;
    }
}
