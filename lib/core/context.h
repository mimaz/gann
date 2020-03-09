#pragma once

#define CL_TARGET_OPENCL_VERSION 100
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include <glib.h>
#include <gio/gio.h>

#define FLAG_DERIVATIVE_INPUT 1

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
    GHashTable *activationtable;
    cl_program built_program;
    cl_program pattern_program;
    cl_kernel pattern_kernel;
    cl_mem pattern_mem;
    int pattern_size;
    int pattern_capacity;
    void *pattern_cache;
    int group_size;
};

struct context *context_create ();
void context_free (struct context *ctx);
const char *context_read_cl_code (struct context *ctx,
                                  const char *name);
void context_add_activation (struct context *ctx,
                             const char *name,
                             const char *code);
void context_program_clear (struct context *ctx);
void context_program_option (struct context *ctx,
                             const char *fmt,
                             ...);
void context_program_activation (struct context *ctx,
                                 const char *name);
void context_program_file (struct context *ctx,
                           const char *name);
void context_program_code (struct context *ctx,
                           const char *src);
void context_program_build (struct context *ctx,
                            cl_program *handle);
void context_program_kernel (struct context *ctx,
                             const char *name,
                             cl_kernel *handle);
void context_fill_pattern (struct context *ctx,
                           cl_mem mem,
                           cl_int memsize,
                           const void *data,
                           cl_int datasize,
                           cl_int evnum,
                           const cl_event *evlist,
                           cl_event *ev);
void context_run_sparse (struct context *ctx,
                         cl_kernel kern,
                         int units,
                         cl_int evcnt,
                         const cl_event *evlist,
                         cl_event *ev);
