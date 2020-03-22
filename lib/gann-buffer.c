/*
 * gann-buffer.c
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

#include "gann-buffer-private.h"
#include "gann-context-private.h"

typedef struct {
    GannContext *context;
    GType element_type;
    gint element_size;
    gint width;
    gint height;
    gint depth;

    gfloat *data;
    cl_mem mem;
    cl_event event;
    cl_int evcount;
    cl_event evlist[8];
} GannBufferPrivate;

enum
{
    PROP_0,
    PROP_CONTEXT,
    PROP_ELEMENT_TYPE,
    PROP_ELEMENT_SIZE,
    PROP_WIDTH,
    PROP_HEIGHT,
    PROP_DEPTH,
    N_PROPS,
};

G_DEFINE_TYPE_WITH_PRIVATE (GannBuffer, gann_buffer,
                            G_TYPE_OBJECT);

static GParamSpec *props[N_PROPS];

static void constructed (GObject *gobj);
static void finalize (GObject *gobj);
static void set_property (GObject *gobj, guint propid,
                          const GValue *value, GParamSpec *spec);
static void get_property (GObject *gobj, guint propid,
                          GValue *value, GParamSpec *spec);

static void
gann_buffer_init (GannBuffer *self)
{
}

static void
gann_buffer_class_init (GannBufferClass *cls)
{
    GObjectClass *gcls = G_OBJECT_CLASS (cls);

    gcls->constructed = constructed;
    gcls->finalize = finalize;
    gcls->set_property = set_property;
    gcls->get_property = get_property;

    props[PROP_CONTEXT] =
        g_param_spec_object ("context",
                             "Context",
                             "Context reference",
                             GANN_TYPE_CONTEXT,
                             G_PARAM_READWRITE |
                             G_PARAM_CONSTRUCT_ONLY |
                             G_PARAM_STATIC_STRINGS);

    props[PROP_ELEMENT_TYPE] =
        g_param_spec_gtype ("element-type",
                            "Element type",
                            "Type of a single element",
                            G_TYPE_NONE,
                            G_PARAM_READWRITE |
                            G_PARAM_CONSTRUCT_ONLY |
                            G_PARAM_STATIC_STRINGS);

    props[PROP_ELEMENT_SIZE] =
        g_param_spec_int ("element-size",
                          "Element size",
                          "Size of a single element",
                          1, 8, 1,
                          G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS);
    
    props[PROP_WIDTH] =
        g_param_spec_int ("width",
                          "Width",
                          "Buffer width dimension",
                          0, G_MAXINT32, 0,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS);

    props[PROP_HEIGHT] =
        g_param_spec_int ("height",
                          "Height",
                          "Buffer height dimension",
                          0, G_MAXINT32, 0,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS);

    props[PROP_DEPTH] =
        g_param_spec_int ("depth",
                          "Depth",
                          "Buffer depth dimension",
                          0, G_MAXINT32, 0,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS);

    g_object_class_install_properties (gcls, N_PROPS, props);
}

static void
constructed (GObject *gobj)
{
    GannBuffer *self = GANN_BUFFER (gobj);
    GannBufferPrivate *p = gann_buffer_get_instance_private (self);

    if (p->element_type == G_TYPE_FLOAT) {
        p->element_size = sizeof (gfloat);
    } else {
        g_error ("invalid buffer element type");
    }

    g_object_notify_by_pspec (gobj, props[PROP_ELEMENT_SIZE]);

    G_OBJECT_CLASS (gann_buffer_parent_class)->constructed (gobj);
}

static void
finalize (GObject *gobj)
{
    GannBuffer *self = GANN_BUFFER (gobj);
    GannBufferPrivate *p = gann_buffer_get_instance_private (self);

    g_clear_pointer (&p->event, clReleaseEvent);
    g_clear_object (&p->context);

    G_OBJECT_CLASS (gann_buffer_parent_class)->finalize (gobj);
}

static void
set_property (GObject *gobj, guint propid,
              const GValue *value, GParamSpec *spec)
{
    GannBuffer *self = GANN_BUFFER (gobj);
    GannBufferPrivate *p = gann_buffer_get_instance_private (self);

    switch (propid) {
    case PROP_CONTEXT:
        g_set_object (&p->context, g_value_get_object (value));
        break;

    case PROP_ELEMENT_TYPE:
        p->element_type = g_value_get_gtype (value);
        break;

    case PROP_WIDTH:
        p->width = g_value_get_int (value);
        break;

    case PROP_HEIGHT:
        p->height = g_value_get_int (value);
        break;

    case PROP_DEPTH:
        p->depth = g_value_get_int (value);
        break;

    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (gobj, propid, spec);
    }
}

static void
get_property (GObject *gobj, guint propid,
              GValue *value, GParamSpec *spec)
{
    GannBuffer *self = GANN_BUFFER (gobj);
    GannBufferPrivate *p = gann_buffer_get_instance_private (self);

    switch (propid) {
    case PROP_CONTEXT:
        g_value_set_object (value, p->context);
        break;

    case PROP_ELEMENT_TYPE:
        g_value_set_gtype (value, p->element_type);
        break;

    case PROP_ELEMENT_SIZE:
        g_value_set_int (value, p->element_size);
        break;

    case PROP_WIDTH:
        g_value_set_int (value, p->width);
        break;

    case PROP_HEIGHT:
        g_value_set_int (value, p->height);
        break;

    case PROP_DEPTH:
        g_value_set_int (value, p->depth);
        break;

    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (gobj, propid, spec);
    }
}

GannBuffer *
gann_buffer_new (GannContext *context,
                 GType element_type,
                 gint width,
                 gint height,
                 gint depth)
{
    return g_object_new (GANN_TYPE_BUFFER,
                         "context", context,
                         "element-type", element_type,
                         "width", width,
                         "height", height,
                         "depth", depth,
                         NULL);
}

/**
 * gann_buffer_get_context:
 *
 * returns: (transfer none):
 */
GannContext *
gann_buffer_get_context (GannBuffer *self)
{
    GannBufferPrivate *p = gann_buffer_get_instance_private (self);

    return p->context;
}

GType
gann_buffer_get_element_type (GannBuffer *self)
{
    GannBufferPrivate *p = gann_buffer_get_instance_private (self);

    return p->element_type;
}

gint
gann_buffer_get_element_size (GannBuffer *self)
{
    GannBufferPrivate *p = gann_buffer_get_instance_private (self);

    return p->element_size;
}

gint
gann_buffer_get_width (GannBuffer *self)
{
    GannBufferPrivate *p = gann_buffer_get_instance_private (self);
    
    return p->width;
}

gint
gann_buffer_get_height (GannBuffer *self)
{
    GannBufferPrivate *p = gann_buffer_get_instance_private (self);
    
    return p->height;
}

gint
gann_buffer_get_depth (GannBuffer *self)
{
    GannBufferPrivate *p = gann_buffer_get_instance_private (self);
    
    return p->depth;
}

/**
 * gann_buffer_write:
 *
 * returns: (transfer none):
 */
GannBuffer *
gann_buffer_write (GannBuffer *self,
                   gint offset,
                   const gfloat *data,
                   gsize size)
{
    GannBufferPrivate *p;
    const cl_event *evlist;

    p = gann_buffer_get_instance_private (self);
    evlist = p->evcount > 0 ? p->evlist : NULL;

    clEnqueueWriteBuffer (gann_context_cl_queue (p->context),
                          p->mem,
                          CL_TRUE,
                          offset * p->element_size,
                          size * p->element_size,
                          data, p->evcount, evlist, &p->event);
    return self;
}

/**
 * gann_buffer_read:
 *
 * returns: (array length=size) (transfer none):
 */
const gfloat *
gann_buffer_read (GannBuffer *self,
                  gint offset,
                  gint count,
                  gsize *size)
{
    GannBufferPrivate *p;
    const cl_event *evlist;

    p = gann_buffer_get_instance_private (self);
    evlist = p->evcount > 0 ? p->evlist : NULL;

    clEnqueueReadBuffer (gann_context_cl_queue (p->context),
                         p->mem,
                         CL_TRUE,
                         offset * p->element_size,
                         count * p->element_size,
                         p->data,
                         p->evcount, evlist, &p->event);

    return p->data;
}

/***************
 * PRIVATE API *
 ***************/

cl_mem
gann_buffer_cl_mem (GannBuffer *self)
{
    GannBufferPrivate *p = gann_buffer_get_instance_private (self);

    return p->mem;
}

cl_event
gann_buffer_cl_event (GannBuffer *self)
{
    GannBufferPrivate *p = gann_buffer_get_instance_private (self);

    return p->event;
}

void
gann_buffer_cl_sync (GannBuffer *self,
                     cl_event barrier, ...)
{
    GannBufferPrivate *p = gann_buffer_get_instance_private (self);
    va_list args;

    va_start (args, barrier);

    p->evcount = 0;

    do {
        g_assert (p->evcount < G_N_ELEMENTS (p->evlist));
        p->evlist[p->evcount++] = barrier;
        barrier = va_arg (args, cl_event);
    } while (barrier != NULL);

    va_end (args);
}
