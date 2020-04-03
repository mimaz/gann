/*
 * gann-output-layer.c
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

#include "gann-output-layer.h"
#include "gann-network.h"
#include "gann-cl-barrier.h"

#include "core/layer.h"

struct _GannOutputLayer
{
    GannLayer parent_instance;
};

static void constructed (GObject *gobj);
static void cl_barrier_init (GannClBarrierInterface *itf);
static gboolean forward_barrier (GannClBarrier *barrier,
                                 cl_event *event);
static gboolean backward_barrier (GannClBarrier *barrier,
                                  cl_event *event);

G_DEFINE_TYPE_WITH_CODE (GannOutputLayer, gann_output_layer,
                         GANN_TYPE_LAYER,
                         G_IMPLEMENT_INTERFACE (GANN_TYPE_CL_BARRIER,
                                                cl_barrier_init));

static void
gann_output_layer_init (GannOutputLayer *self)
{
}

static void
gann_output_layer_class_init (GannOutputLayerClass *cls)
{
    GObjectClass *gcls = G_OBJECT_CLASS (cls);

    gcls->constructed = constructed;
}

static void
constructed (GObject *gobj)
{
    GannLayer *layer;
    GannNetwork *network;
    struct layer *core;

    layer = GANN_LAYER (gobj);
    network = gann_layer_get_network (layer);

    g_assert (gann_layer_get_width (layer) == 0);
    g_assert (gann_layer_get_height (layer) == 0);
    g_assert (gann_layer_get_depth (layer) == 0);

    core = layer_make_output (gann_network_get_core (network));
    gann_layer_set_core (layer, core);

    g_object_set (layer,
                  "width", core->width,
                  "height", core->height,
                  "depth", core->depth,
                  NULL);

    G_OBJECT_CLASS (gann_output_layer_parent_class)->constructed (gobj);
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

GannOutputLayer *
gann_output_layer_new (GannNetwork *network)
{
    return g_object_new (GANN_TYPE_OUTPUT_LAYER,
                         "network", network,
                         NULL);
}

void
gann_output_layer_set_truth (GannOutputLayer *self,
                             const gfloat *data,
                             gsize datasize)
{
    layer_output_set_truth (gann_layer_get_core (GANN_LAYER (self)),
                            data, datasize);
}
