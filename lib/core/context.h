#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <glib.h>
#include <CL/opencl.h>

struct context
{
    GSList *netlist;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
};

struct context *context_create ();
void context_free (struct context *ctx);
