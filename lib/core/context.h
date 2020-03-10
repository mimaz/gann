/*
 * context.h
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

#pragma once

#define CL_TARGET_OPENCL_VERSION 100
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include <glib.h>
#include <gio/gio.h>

struct context
{
    int group_size;

    /* List of network instances */
    GSList *netlist;

    /* Table of OpenCL code where file name is the key */
    GHashTable *codetable;

    /* Table of activation code where name is the key */
    GHashTable *activationtable;

    /* OpenCL code GResource object */
    GResource *resource;

    /* OpenCL context handles */
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    /* Program making variables */
    GString *options;
    GPtrArray *sources;
    cl_program built_program;

#if CL_TARGET_OPENCL_VERSION < 120
    /* Pattern data */
    cl_program pattern_program;
    cl_kernel pattern_kernel;
    cl_mem pattern_mem;
    int pattern_size;
    int pattern_capacity;
    void *pattern_cache;
#endif
};

/*
 * context_create
 * Creates new context
 */
struct context *context_create ();

/*
 * context_free
 * Frees the context
 */
void context_free (struct context *ctx);

/*
 * context_read_cl_code
 * Reads OpenCL resource code
 * name: caller owned string, should be valid only for the call
 * returns: context owned string
 */
const char *context_read_cl_code (struct context *ctx,
                                  const char *name);

/*
 * context_add_activation
 * Reads OpenCL code from GResource
 * name: activation name, owned by the caller and should be
 * valid for the whole context's lifetime
 * code: activation code, owned by the caller and should be
 * valid for the whole context's lifetime
 */
void context_add_activation (struct context *ctx,
                             const char *name,
                             const char *code);

/*
 * context_program_clear
 * Clears program factory
 */
void context_program_clear (struct context *ctx);

/*
 * context_program_option
 * Adds options to program being built
 * fmt: printf-like string format
 * ...: fmt arguments
 */
void context_program_option (struct context *ctx,
                             const char *fmt,
                             ...);

/*
 * context_program_activation
 * Adds activation to program being built
 * name: name of the activation function, owned by the
 * caller at least for the call time
 */
void context_program_activation (struct context *ctx,
                                 const char *name);

/*
 * context_program_file
 * Adds OpenCL source from file
 * name: file name
 */
void context_program_file (struct context *ctx,
                           const char *name);

/*
 * context_program_code
 * Adds given source to the program being built
 * src: string of code, has to be valid at least until
 * program building is finished
 */
void context_program_code (struct context *ctx,
                           const char *src);

/*
 * context_program_build
 * Build program with properties set before,
 * handle: pointer to program handle
 */
void context_program_build (struct context *ctx,
                            cl_program *handle);

/*
 * context_program_kernel
 * Makes kernel for the program built before
 * name: name of the kernel
 * handle: pointer to the kernel handle
 */
void context_program_kernel (struct context *ctx,
                             const char *name,
                             cl_kernel *handle);

/*
 * context_fill_pattern
 * Fills buffer with a pattern. It's workaround for
 * missing clEnqueueFillBuffer in OpenCL 1.0
 * mem: target buffer
 * memsize: fill size
 * data: pattern pointer
 * datasize: pattern size
 * evnum: number of events to the queue
 * evlist: event list to the queue
 * ev: handle to fill task event
 */
void context_fill_pattern (struct context *ctx,
                           cl_mem mem,
                           cl_int memoff,
                           cl_int memsize,
                           const void *data,
                           cl_int datasize,
                           cl_int evnum,
                           const cl_event *evlist,
                           cl_event *ev);

/*
 * context_run_sparse
 * Runs given kernel on as many computation units
 * as possible
 * kern: kernel handle
 * units: unit count
 * evcnt: number of events to the queue
 * evlist: event list to the queue
 * ev: handle to the event
 */
void context_run_sparse (struct context *ctx,
                         cl_kernel kern,
                         int units,
                         cl_int evcnt,
                         const cl_event *evlist,
                         cl_event *ev);
