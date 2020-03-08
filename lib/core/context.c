#include "context.h"
#include "network.h"
#include "cl_code.h"

struct context *
context_create ()
{
    struct context *ctx;
    cl_int err;
    cl_platform_id plat_id;

    ctx = g_new0 (struct context, 1);
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

    ctx->group_size = 256;

    context_program_clear (ctx);

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
    context_program_clear (ctx);

    g_slist_free_full (ctx->netlist, release_network);
    g_slist_free_full (ctx->programlist, release_program);

    clReleaseCommandQueue (ctx->queue);
    clReleaseContext (ctx->context);

    g_hash_table_unref (ctx->codetable);
    g_free (ctx);
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

void
context_program_clear (struct context *ctx)
{
    if (ctx->options != NULL) {
        g_string_free (ctx->options, TRUE);
        ctx->options = NULL;
    }

    g_clear_pointer (&ctx->sources, g_ptr_array_unref);
}

void
context_program_option (struct context *ctx,
                        const char *fmt,
                        ...)
{
    va_list args;

    if (ctx->options == NULL) {
        ctx->options = g_string_new (NULL);
    }

    va_start (args, fmt);
    g_string_append_vprintf (ctx->options, fmt, args);
    g_string_append_c (ctx->options, ' ');
    va_end (args);
}

void
context_program_file (struct context *ctx,
                      const char *name)
{
    const char *code;

    code = context_read_cl_code (ctx, name);

    context_program_code (ctx, code);
}

void
context_program_code (struct context *ctx,
                      const char *code)
{
    if (ctx->sources == NULL) {
        ctx->sources = g_ptr_array_new_with_free_func (g_free);
    }

    g_ptr_array_insert (ctx->sources, -1, g_strdup (code));
}

cl_program
context_program_build (struct context *ctx)
{
    cl_program prog;
    cl_int err;
    char *log;
    size_t logsize;

    prog = clCreateProgramWithSource (ctx->context,
                                      ctx->sources->len,
                                      (const char **)
                                      ctx->sources->pdata,
                                      NULL, &err);

    g_assert (err == CL_SUCCESS);

    err = clBuildProgram (prog, 0, NULL, ctx->options->str, NULL, NULL);

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

    context_program_clear (ctx);

    return prog;
}
