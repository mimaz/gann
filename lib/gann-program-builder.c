/*
 * gann-program-builder.h
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

#include "gann-program-builder.h"

#include "gann-context.h"
#include "gann-context-private.h"

struct _GannProgramBuilder
{
    GObject parent_instance;
    GannContext *context;
    GPtrArray *src_arr;
    GHashTable *kern_table;
    GSList *prog_list;
    GString *options;
};

enum
{
    PROP_0,
    PROP_CONTEXT,
    N_PROPS,
};

G_DEFINE_TYPE (GannProgramBuilder, gann_program_builder, G_TYPE_OBJECT);

static GParamSpec *props[N_PROPS];

static void set_property (GObject *gobj, guint propid,
                          const GValue *value, GParamSpec *spec);
static void get_property (GObject *gobj, guint propid,
                          GValue *value, GParamSpec *spec);
static void dispose (GObject *gobj);
static void finalize (GObject *gobj);

static void
gann_program_builder_init (GannProgramBuilder *self)
{
    self->src_arr = g_ptr_array_new ();
    self->kern_table = g_hash_table_new (g_str_hash, g_str_equal);
    self->prog_list = NULL;
    self->options = g_string_new (NULL);
}

static void
gann_program_builder_class_init (GannProgramBuilderClass *cls)
{
    GObjectClass *gcls = G_OBJECT_CLASS (cls);

    gcls->set_property = set_property;
    gcls->get_property = get_property;
    gcls->dispose = dispose;
    gcls->finalize = finalize;

    props[PROP_CONTEXT] =
        g_param_spec_object ("context",
                             "Context",
                             "Context reference",
                             GANN_TYPE_CONTEXT,
                             G_PARAM_WRITABLE |
                             G_PARAM_CONSTRUCT_ONLY |
                             G_PARAM_STATIC_STRINGS);

    g_object_class_install_properties (gcls, N_PROPS, props);
}

static void
set_property (GObject *gobj,
              guint propid,
              const GValue *value,
              GParamSpec *spec)
{
    GannProgramBuilder *self = GANN_PROGRAM_BUILDER (gobj);

    switch (propid) {
    case PROP_CONTEXT:
        g_set_object (&self->context, g_value_get_object (value));
        break;
        
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (gobj, propid, spec);
    }
}

static void
get_property (GObject *gobj,
              guint propid,
              GValue *value,
              GParamSpec *spec)
{
    switch (propid) {
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (gobj, propid, spec);
    }
}

static void
dispose (GObject *gobj)
{
    GannProgramBuilder *self = GANN_PROGRAM_BUILDER (gobj);

    g_clear_object (&self->context);

    G_OBJECT_CLASS (gann_program_builder_parent_class)->dispose (gobj);
}

static void finalize (GObject *gobj)
{
    GannProgramBuilder *self = GANN_PROGRAM_BUILDER (gobj);

    g_ptr_array_unref (self->src_arr);
    g_hash_table_unref (self->kern_table);
    g_string_free (self->options, TRUE);

    G_OBJECT_CLASS (gann_program_builder_parent_class)->finalize (gobj);
}

GannProgramBuilder *
gann_program_builder_new (GannContext *context)
{
    return g_object_new (GANN_TYPE_PROGRAM_BUILDER,
                         "context", context,
                         NULL);
}

void
gann_program_builder_clear (GannProgramBuilder *self)
{
    g_ptr_array_set_size (self->src_arr, 0);
    g_hash_table_remove_all (self->kern_table);
    g_slist_free (self->prog_list);
    self->prog_list = NULL;
    g_string_assign (self->options, "");
}

void
gann_program_builder_option (GannProgramBuilder *self,
                             const gchar *fmt, ...)
{
    va_list args;

    va_start (args, fmt);

    if (self->options->len > 0) {
        g_string_append_c (self->options, ' ');
    }
    g_string_append_vprintf (self->options, fmt, args);

    va_end (args);
}

void
gann_program_builder_activation (GannProgramBuilder *self,
                                 gint index,
                                 const gchar *name)
{
    const gchar *code;

    code = gann_context_activation (self->context, name);

    gann_program_builder_code (self, code);
}

void
gann_program_builder_file (GannProgramBuilder *self,
                           const gchar *name)
{
    const gchar *code;

    code = gann_context_code (self->context, name);

    gann_program_builder_code (self, code);
}

void
gann_program_builder_code (GannProgramBuilder *self,
                           const gchar *code)
{
    g_ptr_array_insert (self->src_arr, -1, (gpointer) code);
}

void
gann_program_builder_program (GannProgramBuilder *self,
                              cl_program *handle)
{
    self->prog_list = g_slist_prepend (self->prog_list, handle);
}

void
gann_program_builder_kernel (GannProgramBuilder *self,
                             const gchar *name,
                             cl_kernel *handle)
{
    g_hash_table_insert(self->kern_table, (gpointer) name, handle);
}

void
gann_program_builder_build (GannProgramBuilder *self)
{
    cl_context clctx;
    cl_device_id cldev;
    cl_int err;
    cl_program prog;
    size_t logsize;
    gchar *logstr;
    GSList *progit;
    GHashTableIter kernit;
    gpointer kernptr, nameptr;

    clctx = gann_context_cl_context (self->context);
    cldev = gann_context_cl_device (self->context);

    prog = clCreateProgramWithSource (clctx,
                                      self->src_arr->len,
                                      (const gchar **)
                                      self->src_arr->pdata,
                                      NULL, &err);
    g_assert (err == CL_SUCCESS);

    err = clBuildProgram (prog, 0, NULL,
                          self->options->str,
                          NULL, NULL);

    if (err != CL_SUCCESS) {
        clGetProgramBuildInfo (prog, cldev,
                               CL_PROGRAM_BUILD_LOG,
                               0, NULL, &logsize);
        logstr = g_new (gchar, logsize);
        clGetProgramBuildInfo (prog, cldev,
                               CL_PROGRAM_BUILD_LOG,
                               logsize, logstr, NULL);
        g_error (logstr);
        g_free (logstr);
    }

    progit = self->prog_list;

    while (progit != NULL) {
        *(cl_program *) progit->data = prog;
        progit = progit->next;
    }

    g_hash_table_iter_init (&kernit, self->kern_table);

    while (g_hash_table_iter_next (&kernit, &kernptr, &nameptr)) {
        *(cl_kernel *) kernptr = clCreateKernel (prog, nameptr, &err);
        g_assert (err == CL_SUCCESS);
    }
}
