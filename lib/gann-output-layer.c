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

struct _GannOutputLayer
{
    GannLayer parent_instance;
	GannBuffer *truth_buffer;
};

static void constructed (GObject *gobj);
static void compile (GannLayer *layer);
static void forward (GannLayer *layer);
static GannBuffer *value_buffer (GannLayer *self);
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
	GannLayerClass *lcls = GANN_LAYER_CLASS (cls);

    gcls->constructed = constructed;
	lcls->compile = compile;
	lcls->forward = forward;
	lcls->value_buffer = value_buffer;
}

static void
constructed (GObject *gobj)
{
    G_OBJECT_CLASS (gann_output_layer_parent_class)->constructed (gobj);
}

static void
compile (GannLayer *layer)
{
	GannOutputLayer *self;
	GannLayer *prev;

	self = GANN_OUTPUT_LAYER (layer);
	prev = gann_layer_prev_layer (layer);

	g_object_set (layer,
				  "width", gann_layer_get_width (prev),
				  "height", gann_layer_get_height (prev),
				  "depth", gann_layer_get_depth (prev),
				  NULL);

	self->truth_buffer = gann_buffer_new (gann_layer_get_context (layer),
										  G_TYPE_FLOAT,
										  gann_layer_get_height (layer),
										  gann_layer_get_width (layer),
										  gann_layer_get_depth (layer));

	GANN_LAYER_CLASS (gann_output_layer_parent_class)->compile (layer);
}

static void
forward (GannLayer *layer)
{
	GannLayer *prev;

	prev = gann_layer_prev_layer (layer);

	GANN_LAYER_CLASS (gann_output_layer_parent_class)->forward (layer);
}

static GannBuffer *
value_buffer (GannLayer *layer)
{
	GannLayer *prev;

	prev = gann_layer_prev_layer (layer);

	return gann_layer_value_buffer (prev);
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
	gann_buffer_write (self->truth_buffer,
					   0, data, datasize);
}
