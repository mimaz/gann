#include "context.h"
#include "network.h"
#include "cl_code.h"

struct context *
context_create ()
{
    struct context *ctx;
    cl_int err;
    cl_platform_id plat_id;

    ctx = g_new (struct context, 1);
    ctx->netlist = NULL;
    ctx->cltable = g_hash_table_new_full (g_str_hash,
                                          g_str_equal,
                                          g_free,
                                          (GDestroyNotify)
                                          g_bytes_unref);
    ctx->resource = cl_code_get_resource ();

    err = clGetPlatformIDs (1, &plat_id, NULL);
    g_assert (err == 0);

    err = clGetDeviceIDs (plat_id, CL_DEVICE_TYPE_GPU, 1, &ctx->device, NULL);
    g_assert (err == 0);

    ctx->context = clCreateContext (0, 1, &ctx->device, NULL, NULL, &err);
    g_assert (err == 0);

    ctx->queue = clCreateCommandQueue (ctx->context, ctx->device, 0, &err);
    g_assert (err == 0);

    return ctx;
}

void
context_free (struct context *ctx)
{
    clReleaseCommandQueue (ctx->queue);
    clReleaseContext (ctx->context);

    g_slist_free_full (ctx->netlist, (GDestroyNotify) network_free);
    g_hash_table_unref (ctx->cltable);
    g_free (ctx);
}

const char *
context_read_cl_code (struct context *ctx,
                      const char *name)
{
    char *path;
    GBytes *bytes;

    bytes = g_hash_table_lookup (ctx->cltable, name);

    if (bytes == NULL) {
        path = g_strdup_printf ("/gann/core/cl/%s", name);
        bytes = g_resource_lookup_data (ctx->resource,
                                        path,
                                        G_RESOURCE_FLAGS_NONE,
                                        NULL);
        g_assert (bytes != NULL);
        g_hash_table_insert (ctx->cltable,
                             g_strdup (name),
                             bytes);
    }

    return g_bytes_get_data (bytes, NULL);
}
