/*
 * gann-context.c
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

#include "gann-context.h"

#include "core/core.h"

struct _GannContext
{
    GObject parent_instance;
    struct context *core;
    GSList *networks;
};

G_DEFINE_TYPE (GannContext, gann_context, G_TYPE_OBJECT);

static void dispose (GObject *gobj);
static void constructed (GObject *gobj);

static void
gann_context_init (GannContext *self)
{
    self->networks = NULL;
}

static void
gann_context_class_init (GannContextClass *cls)
{
    GObjectClass *gcls = G_OBJECT_CLASS (cls);

    gcls->dispose = dispose;
    gcls->constructed = constructed;
}

static void
dispose (GObject *gobj)
{
    GannContext *self = GANN_CONTEXT (gobj);

    g_clear_pointer (&self->core, context_free);

    G_OBJECT_CLASS (gann_context_parent_class)->dispose (gobj);
}

static void
constructed (GObject *gobj)
{
    GannContext *self = GANN_CONTEXT (gobj);

    self->core = context_create ();

    G_OBJECT_CLASS (gann_context_parent_class)->constructed (gobj);
}

GannContext *
gann_context_new ()
{
    return g_object_new (GANN_TYPE_CONTEXT, NULL);
}

void
gann_context_add_network (GannContext *self,
                          GannNetwork *network)
{
    self->networks = g_slist_append (self->networks, network);
}

void
gann_context_remove_network (GannContext *self,
                             GannNetwork *network)
{
    self->networks = g_slist_remove (self->networks, network);
}

struct context *
gann_context_get_core (GannContext *self)
{
    return self->core;
}

/***************
 * PRIVATE API *
 ***************/

cl_command_queue
gann_context_cl_queue (GannContext *self)
{
    return self->core->queue;
}

cl_context
gann_context_cl_context (GannContext *self)
{
    return self->core->context;
}

cl_device_id
gann_context_cl_device (GannContext *self)
{
    return self->core->device;
}

const gchar *
gann_context_code (GannContext *self,
                   const gchar *filename)
{
    return context_read_cl_code (self->core, filename);
}

const gchar *
gann_context_activation (GannContext *self,
                         const gchar *name)
{
    return name;
}
