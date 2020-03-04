#include "layer.h"
#include "network.h"

#include <math.h>

void
layer_free (struct layer *lay)
{
    int i;

    if (lay->release != NULL) {
        lay->release (lay);
    }

    for (i = 0; i < G_N_ELEMENTS (lay->kernels); i++) {
        clReleaseKernel (lay->kernels[i]);
    }

    g_clear_pointer (&lay->value_v, g_free);
    g_clear_pointer (&lay->gradient_v, g_free);
    g_clear_pointer (&lay->weight_v, g_free);
    g_clear_pointer (&lay->delta_v, g_free);

    g_free (lay);
}

cl_kernel
layer_create_kernel (struct layer *lay,
                     int id, const char *name)
{
    cl_int err;
    lay->kernels[id] = clCreateKernel (lay->program, name, &err);
    g_assert (err == CL_SUCCESS);
    return lay->kernels[id];
}
