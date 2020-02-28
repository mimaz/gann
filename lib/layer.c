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
    g_clear_pointer (&lay->gradient_v, g_free);
    g_clear_pointer (&lay->weight_v, g_free);
    g_clear_pointer (&lay->delta_v, g_free);

    g_free (lay);
}
