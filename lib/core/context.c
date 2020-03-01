#include "context.h"
#include "network.h"

struct context *
context_create ()
{
    struct context *ctx;
    cl_int err;
    cl_platform_id plat_id;
    cl_device_id dev_id;

    ctx = g_new (struct context, 1);
    ctx->netlist = NULL;

    err = clGetPlatformIDs (1, &plat_id, NULL);
    g_assert (err == 0);

    err = clGetDeviceIDs (plat_id, CL_DEVICE_TYPE_GPU, 1, &dev_id, NULL);
    g_assert (err == 0);

    ctx->context = clCreateContext (0, 1, &dev_id, NULL, NULL, &err);
    g_assert (err == 0);

    ctx->queue = clCreateCommandQueue (ctx->context, dev_id, 0, &err);
    g_assert (err == 0);

    return ctx;
}

void
context_free (struct context *ctx)
{
    clReleaseCommandQueue (ctx->queue);
    clReleaseContext (ctx->context);

    g_slist_free_full (ctx->netlist, (GDestroyNotify) network_free);
    g_free (ctx);
}
