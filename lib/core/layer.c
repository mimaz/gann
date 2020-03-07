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
    g_assert (lay->value_mem != 0);
    g_assert (offset + count <= lay->size);
    clEnqueueReadBuffer (lay->net->ctx->queue,
                         lay->value_mem,
                         CL_TRUE,
                         offset * sizeof (cl_float),
                         count * sizeof (cl_float),
                         buff, 0, NULL, NULL);
}

void
layer_create_kernel (struct layer *lay,
                     cl_kernel *handle,
                     const char *name)
{
    cl_kernel kern;
    cl_int err;

    kern = clCreateKernel (lay->program, name, &err);
    g_assert (err == CL_SUCCESS);

    *handle = kern;
}

void
layer_create_buffer (struct layer *lay,
                     cl_mem *handle,
                     int size,
                     int flags)
{
    cl_int err;
    cl_mem mem;

    mem = clCreateBuffer (lay->net->ctx->context,
                          flags,
                          size * sizeof (cl_float),
                          NULL, &err);

    g_assert (err == CL_SUCCESS);

    *handle = mem;
}
