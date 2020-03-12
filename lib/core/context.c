/*
 * context.c
 *
 * Copyright 2020 Mieszko Mazurek <mimaz@gmx.com>
 *
 * This file is part of Gann.
 *
 * Gann is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Gann is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Gann.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "context.h"
#include "network.h"
#include "util.h"
#include "cl_code.h"

static void
add_activation_from_source (struct context *ctx,
                            const char *name,
                            const char *file)
{
    const char *code;

    code = context_read_cl_code (ctx, file);

    context_add_activation (ctx, name, code);
}

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
                                            NULL,
                                            (GDestroyNotify)
                                            g_bytes_unref);
    ctx->resource = cl_code_get_resource ();
    ctx->activationtable = g_hash_table_new (g_str_hash,
                                             g_str_equal);

    err = clGetPlatformIDs (1, &plat_id, NULL);
    g_assert (err == 0);

    err = clGetDeviceIDs (plat_id, CL_DEVICE_TYPE_CPU, 1, &ctx->device, NULL);
    g_assert (err == 0);

    ctx->context = clCreateContext (0, 1, &ctx->device, NULL, NULL, &err);
    g_assert (err == 0);

    ctx->queue = clCreateCommandQueue (ctx->context, ctx->device, 0, &err);
    g_assert (err == 0);

    ctx->group_size = 32;

    context_program_clear (ctx);
    context_program_file (ctx, "clear-buffer.cl");
    context_program_build (ctx, &ctx->clear_program);
    context_program_kernel (ctx, "clear", &ctx->clear_kernel);

    add_activation_from_source (ctx, "sigmoid", "sigmoid.cl");
    add_activation_from_source (ctx, "softplus", "softplus.cl");
    add_activation_from_source (ctx, "relu", "relu.cl");
    add_activation_from_source (ctx, "leaky", "leaky.cl");

    return ctx;
}

static void
release_network (gpointer data, gpointer user_data)
{
    (void) user_data;
    network_free (data);
}

void
context_free (struct context *ctx)
{
    GSList *nets;

    nets = g_slist_copy (ctx->netlist);

    g_slist_foreach (nets, release_network, NULL);
    g_slist_free (nets);
    g_assert_null (ctx->netlist);
    g_hash_table_unref (ctx->codetable);
    g_hash_table_unref (ctx->activationtable);

    clReleaseCommandQueue (ctx->queue);
    clReleaseContext (ctx->context);

    /* TODO do we need to release ctx->device? */

    if (ctx->options != NULL) {
        g_string_free (ctx->options, TRUE);
    }

    g_clear_pointer (&ctx->sources, g_ptr_array_unref);
    context_program_clear (ctx);

    /* No need to release ctx->built_program, it's weak handle */

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
        g_free (path);

        /* bytes is now owned by the codetable */
        g_hash_table_insert (ctx->codetable, (gpointer) name, bytes);
    }

    return g_bytes_get_data (bytes, NULL);
}

void
context_add_activation (struct context *ctx,
                        const char *name,
                        const char *code)
{
    g_hash_table_insert (ctx->activationtable,
                         (gpointer) name,
                         (gpointer) code);
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
context_program_activation (struct context *ctx,
                            const char *name)
{
    const char *code;

    code = g_hash_table_lookup (ctx->activationtable, name);
    g_assert (code != NULL);

    context_program_code (ctx, code);
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

void
context_program_build (struct context *ctx,
                       cl_program *handle)
{
    size_t logsize;
    char *log;
    cl_program prog;
    cl_int err;

    prog = clCreateProgramWithSource (ctx->context,
                                      ctx->sources->len,
                                      (const char **)
                                      ctx->sources->pdata,
                                      NULL, &err);

    g_assert (err == CL_SUCCESS);

    err = clBuildProgram (prog, 0, NULL,
                          ctx->options != NULL ? ctx->options->str : NULL,
                          NULL, NULL);

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

    context_program_clear (ctx);

    ctx->built_program = prog;
    *handle = prog;
}

void
context_program_kernel (struct context *ctx,
                        const char *name,
                        cl_kernel *handle)
{
    cl_kernel kern;
    cl_int err;

    kern = clCreateKernel (ctx->built_program, name, &err);
    g_assert (err == CL_SUCCESS);

    *handle = kern;
}

void
context_clear_buffer (struct context *ctx,
                      cl_mem mem,
                      cl_int size,
                      cl_event *ev)
{
    clSetKernelArg (ctx->clear_kernel, 0, sizeof (cl_mem), &mem);
    clSetKernelArg (ctx->clear_kernel, 1, sizeof (cl_int), &size);
    context_run_sparse (ctx, ctx->clear_kernel, size,
                        0, NULL, ev);
}

void
context_run_sparse (struct context *ctx,
                    cl_kernel kern,
                    int units,
                    cl_int evcnt,
                    const cl_event *evlist,
                    cl_event *ev)
{
    size_t globsize, locsize;
    cl_int err;

    locsize = MIN (ctx->group_size, units);
    globsize = util_upper_multiply (units, locsize);

    err = clEnqueueNDRangeKernel (ctx->queue,
                                  kern, 1, NULL,
                                  &globsize, &locsize,
                                  evcnt, evlist, ev);
    g_assert (err == CL_SUCCESS);
}
