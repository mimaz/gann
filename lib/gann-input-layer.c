/*
 * gann-input-layer.c
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

#include "gann-input-layer.h"
#include "gann-network.h"
#include "gann-cl-barrier.h"

struct _GannInputLayer
{
    GObject parent_instance;
};

static void dispose (GObject *gobj);
static void constructed (GObject *gobj);
static void finalize (GObject *gobj);
static void forward (GannLayer *layer);
static void backward (GannLayer *layer);
static void compile (GannLayer *layer);
static void cl_barrier_init (GannClBarrierInterface *itf);
static gboolean forward_barrier (GannClBarrier *barrier,
                                 cl_event *event);
static gboolean backward_barrier (GannClBarrier *barrier,
                                  cl_event *event);

G_DEFINE_TYPE_WITH_CODE (GannInputLayer, gann_input_layer,
                         GANN_TYPE_LAYER,
                         G_IMPLEMENT_INTERFACE (GANN_TYPE_CL_BARRIER,
                                                cl_barrier_init));

static void
gann_input_layer_init (GannInputLayer *self)
{
}

static void
gann_input_layer_class_init (GannInputLayerClass *cls)
{
    GObjectClass *gcls = G_OBJECT_CLASS (cls);
    GannLayerClass *lcls = GANN_LAYER_CLASS (cls);

    gcls->dispose = dispose;
    gcls->constructed = constructed;
    gcls->finalize = finalize;

    lcls->forward = forward;
    lcls->backward = backward;
    lcls->compile = compile;
}

static void
dispose (GObject *gobj)
{
    /* allocated data block is freed by core layer */
    G_OBJECT_CLASS (gann_input_layer_parent_class)->dispose (gobj);
}

static void
constructed (GObject *gobj)
{
    G_OBJECT_CLASS (gann_input_layer_parent_class)->constructed (gobj);
}

static void
finalize (GObject *gobj)
{
    G_OBJECT_CLASS (gann_input_layer_parent_class)->finalize (gobj);
}

static void
forward (GannLayer *layer)
{
    GANN_LAYER_CLASS (gann_input_layer_parent_class)->forward (layer);
}

static void
backward (GannLayer *layer)
{
}

static void
compile (GannLayer *layer)
{
	GANN_LAYER_CLASS (gann_input_layer_parent_class)->compile (layer);
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

/**
 * gann_input_layer_new:
 *
 * returns: (transfer full): New input layer instance
 */
GannInputLayer *
gann_input_layer_new (GannNetwork *network,
                      gint width,
                      gint height,
                      gint depth)
{
    return g_object_new (GANN_TYPE_INPUT_LAYER,
                         "network", network,
                         "width", width,
                         "height", height,
                         "depth", depth,
                         NULL);
}

/**
 * gann_input_layer_set_data:
 * @data: (array length=size): float array
 */
void
gann_input_layer_set_data (GannInputLayer *self,
                           const gfloat *data,
                           gint size)
{
    GannLayer *layer;
	GannBuffer *buff;

    layer = GANN_LAYER (self);
	buff = gann_layer_value_buffer (layer);
    g_assert (size == gann_layer_get_size (layer));

	gann_buffer_write (buff, 0, data, size);
}

/**
 * gann_input_layer_set_data_bytes:
 * @data: (array length=size): byte array
 */
void
gann_input_layer_set_data_bytes (GannInputLayer *self,
                                 const guint8 *data,
                                 gint size)
{
    gfloat *floats;
    gsize i;

    floats = g_newa (gfloat, size);

    for (i = 0; i < size; i++) {
        floats[i] = data[i] / 255.0f;
    }

    gann_input_layer_set_data (self, floats, size);
}
