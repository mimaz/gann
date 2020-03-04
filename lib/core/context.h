#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <glib.h>
#include <gio/gio.h>
#include <CL/opencl.h>

struct context
{
    GSList *netlist;
    GHashTable *cltable;
    GResource *resource;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
};

struct context *context_create ();
void context_free (struct context *ctx);
const char *context_read_cl_code (struct context *ctx,
                                  const char *name);
