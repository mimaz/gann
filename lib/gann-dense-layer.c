/*
 * gann-dense-layer.c
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

#include "gann-dense-layer.h"
#include "gann-network.h"
#include "gann-cl-barrier.h"

struct _GannDenseLayer
{
    GannLayer parent_instance;
};

static void constructed (GObject *gobj);
static void cl_barrier_init (GannClBarrierInterface *itf);
static gboolean forward_barrier (GannClBarrier *barrier,
                                 cl_event *event);
static gboolean backward_barrier (GannClBarrier *barrier,
                                  cl_event *event);

G_DEFINE_TYPE_WITH_CODE (GannDenseLayer, gann_dense_layer,
                         GANN_TYPE_LAYER,
                         G_IMPLEMENT_INTERFACE (GANN_TYPE_CL_BARRIER,
                                                cl_barrier_init));

static void
gann_dense_layer_init (GannDenseLayer *self)
{
}

static void
gann_dense_layer_class_init (GannDenseLayerClass *cls)
{
    GObjectClass *gcls = G_OBJECT_CLASS (cls);

    gcls->constructed = constructed;
}

static void
constructed (GObject *gobj)
{
    GannLayer *layer;
    GannNetwork *network;

    layer = GANN_LAYER (gobj);
    network = gann_layer_get_network (layer);
}

static void
cl_barrier_init (GannClBarrierInterface *itf)
{
    itf->forward_barrier = forward_barrier;
    itf->backward_barrier = backward_barrier;
}

static gboolean
forward_barrier (GannClBarrier *barrier,
                 cl_event *event)
{
    return FALSE;
}

static gboolean
backward_barrier (GannClBarrier *barrier,
                  cl_event *event)
{
    return FALSE;
}

GannDenseLayer *
gann_dense_layer_new (GannNetwork *network,
                      gint width,
                      gint height,
                      gint depth,
                      const gchar *activation)
{
    return g_object_new (GANN_TYPE_DENSE_LAYER,
                         "network", network,
                         "width", width,
                         "height", height,
                         "depth", depth,
                         "activation", activation,
                         NULL);
}
