#include "layer.h"
#include "network.h"

#include <math.h>

void
layer_free (struct layer *lay)
{
    if (lay->release != NULL) {
        lay->release (lay);
    }

    g_free (lay);
}

void
layer_load_value (struct layer *lay,
                  float *buff,
                  int offset,
                  int count)
{
    clEnqueueReadBuffer (lay->net->ctx->queue,
                         lay->value_mem,
                         CL_TRUE,
                         offset * sizeof (cl_float),
                         count * sizeof (cl_float),
                         buff, 0, NULL, NULL);
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
