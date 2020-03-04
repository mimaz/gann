#pragma once

#define CL_TARGET_OPENCL_VERSION 100

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
};

struct context *context_create ();
void context_free (struct context *ctx);
const char *context_read_cl_code (struct context *ctx,
                                  const char *name);
cl_program context_build_program (struct context *ctx,
                                  const char *options,
                                  const char *firstfile,
                                  ...);
