/*
 * gann-barrier.c
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

#include "gann-barrier-private.h"
#include "gann-context.h"

struct _GannBarrier
{
    GObject parent_instance;

    GannContext *context;
    GPtrArray *events;
};

enum
{
    PROP_0,
    PROP_CONTEXT,
    N_PROPS,
};

G_DEFINE_TYPE (GannBarrier, gann_barrier, G_TYPE_OBJECT);

static GParamSpec *props[N_PROPS];

static void set_property (GObject *gobj, guint propid,
                          const GValue *value, GParamSpec *spec);
static void get_property (GObject *gobj, guint propid,
                          GValue *value, GParamSpec *spec);
static void finalize (GObject *gobj);
static void constructed (GObject *gobj);

static void
gann_barrier_init (GannBarrier *self)
{
}

static void
gann_barrier_class_init (GannBarrierClass *cls)
{
    GObjectClass *gcls = G_OBJECT_CLASS (cls);

    gcls->set_property = set_property;
    gcls->get_property = get_property;
    gcls->finalize = finalize;
    gcls->constructed = constructed;

    props[PROP_CONTEXT] =
        g_param_spec_object ("context",
                             "Context",
                             "Context reference",
                             GANN_TYPE_CONTEXT,
                             G_PARAM_READWRITE |
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
    GannBarrier *self = GANN_BARRIER (gobj);

    switch (propid) {
    case PROP_CONTEXT:
        self->context = g_value_get_object (value);
        g_object_ref (self->context);
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
    GannBarrier *self = GANN_BARRIER (gobj);

    switch (propid) {
    case PROP_CONTEXT:
        g_value_set_object (value, self->context);
        break;

    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (gobj, propid, spec);
    }
}

static void
finalize (GObject *gobj)
{
    GannBarrier *self = GANN_BARRIER (gobj);

    g_clear_object (&self->context);
    g_clear_pointer (&self->events, g_ptr_array_unref);

    G_OBJECT_CLASS (gann_barrier_parent_class)->finalize (gobj);
}

static void
constructed (GObject *gobj)
{
    GannBarrier *self = GANN_BARRIER (gobj);

    self->events = g_ptr_array_new_with_free_func ((GDestroyNotify)
                                                   clReleaseEvent);

    G_OBJECT_CLASS (gann_barrier_parent_class)->constructed (gobj);
}

/**
 * gann_barrier_new:
 *
 * returns: (transfer full):
 */
GannBarrier *
gann_barrier_new (GannContext *context)
{
    return g_object_new (GANN_TYPE_BARRIER,
                         "context", context,
                         NULL);
}

/**
 * gann_barrier_new_from_barrier:
 * 
 * returns: (transfer full):
 */
GannBarrier *
gann_barrier_new_from_barrier (GannBarrier *barrier)
{
    GannBarrier *self;

    self = g_object_new (GANN_TYPE_BARRIER,
                         "context", gann_barrier_get_context (barrier),
                         NULL);

    gann_barrier_attach (self, barrier);

    return self;
}

/**
 * gann_barrier_get_context:
 *
 * returns: (transfer none):
 */
GannContext *
gann_barrier_get_context (GannBarrier *self)
{
    return self->context;
}

void
gann_barrier_attach (GannBarrier *self,
                     GannBarrier *barrier)
{
    const cl_event *list;
    gint count, i;

    list = gann_barrier_cl_events (barrier, &count);

    for (i = 0; i < count; i++) {
        gann_barrier_add_cl_event (self, list[i]);
    }
}

/***************
 * PRIVATE API *
 ***************/

void
gann_barrier_add_cl_event (GannBarrier *self,
                           cl_event event)
{
    g_ptr_array_insert (self->events, -1, event);
}

void
gann_barrier_remove_cl_event (GannBarrier *self,
                              cl_event event)
{
    g_ptr_array_remove (self->events, event);
}

void
gann_barrier_cl_clear (GannBarrier *self)
{
    g_ptr_array_unref (self->events);
    self->events = g_ptr_array_new_with_free_func ((GDestroyNotify)
                                                   clReleaseEvent);
}

const cl_event *
gann_barrier_cl_events (GannBarrier *self,
                        gint *count)
{
    if (count != NULL) {
        *count = self->events->len;
    }

    return (const cl_event *) self->events->pdata;
}
