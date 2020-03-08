#pragma once

#define CL_TARGET_OPENCL_VERSION 100
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <glib.h>
#include <gio/gio.h>
#include <CL/cl.h>

struct context
{
    GSList *netlist;
    GHashTable *codetable;
    GSList *programlist;
    GResource *resource;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    GString *options;
    GPtrArray *sources;
    int group_size;
};

struct context *context_create ();
void context_free (struct context *ctx);
const char *context_read_cl_code (struct context *ctx,
                                  const char *name);
void context_program_clear (struct context *ctx);
void context_program_option (struct context *ctx,
                             const char *fmt,
                             ...);
void context_program_file (struct context *ctx,
                           const char *name);
void context_program_code (struct context *ctx,
                             const char *src);
cl_program context_program_build (struct context *ctx);
