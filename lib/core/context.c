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
    ctx->codetable = g_hash_table_new_full (g_str_hash,
                                          g_str_equal,
                                          g_free,
                                          (GDestroyNotify)
                                          g_bytes_unref);
    ctx->programlist = NULL;
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

static void
release_network (gpointer net)
{
    network_free (net);
}

static void
release_program (gpointer prog)
{
    clReleaseProgram ((cl_program) prog);
}

void
context_free (struct context *ctx)
{
    g_message ("context_free");
    g_slist_free_full (ctx->netlist, release_network);
    g_message ("context_free");
    g_slist_free_full (ctx->programlist, release_program);
    g_message ("context_free");

    clReleaseCommandQueue (ctx->queue);
    g_message ("context_free");
    clReleaseContext (ctx->context);
    g_message ("context_free");

    g_hash_table_unref (ctx->codetable);
    g_message ("context_free");
    g_free (ctx);
    g_message ("context_free");
}

const char *
context_read_cl_code (struct context *ctx,
                      const char *name)
{
    char *path;
    GBytes *bytes;

    bytes = g_hash_table_lookup (ctx->codetable, name);

    if (bytes == NULL) {
        path = g_strdup_printf ("/gann/core/cl/%s", name);
        bytes = g_resource_lookup_data (ctx->resource,
                                        path,
                                        G_RESOURCE_FLAGS_NONE,
                                        NULL);
        g_assert (bytes != NULL);
        g_hash_table_insert (ctx->codetable,
                             g_strdup (name),
                             bytes);
    }

    return g_bytes_get_data (bytes, NULL);
}

cl_program
context_build_program (struct context *ctx,
                       const char *options,
                       const char *firstfile,
                       ...)
{
    cl_program prog;
    cl_int err;
    char *log;
    size_t logsize;
    g_autoptr (GPtrArray) arr;
    va_list args;
    const char *src;

    arr = g_ptr_array_new ();

    va_start (args, firstfile);

    do {
        src = context_read_cl_code (ctx, firstfile);
        g_ptr_array_insert (arr, -1, (gpointer) src);
        firstfile = va_arg (args, const char *);
    } while (firstfile);

    va_end (args);

    prog = clCreateProgramWithSource (ctx->context,
                                      arr->len,
                                      (const char **) arr->pdata,
                                      NULL, &err);

    g_assert (err == CL_SUCCESS);

    err = clBuildProgram (prog, 0, NULL, options, NULL, NULL);

    if (err != CL_SUCCESS) {
        clGetProgramBuildInfo (prog, ctx->device,
                               CL_PROGRAM_BUILD_LOG,
                               0, NULL, &logsize);
        log = g_new (char, logsize);
        clGetProgramBuildInfo (prog, ctx->device,
                               CL_PROGRAM_BUILD_LOG,
                               logsize, log, NULL);
        g_error (log);
        g_free (log);
    }

    ctx->programlist = g_slist_prepend (ctx->programlist, prog);

    return prog;
}
