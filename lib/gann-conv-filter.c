/*
 * gann-conv-filter.h
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

#include "gann-conv-filter.h"

struct _GannConvFilter
{
    GObject parent_instance;
    gint width;
    gint height;
    gint depth;
    gint size;
    gfloat *value_v;
};

G_DEFINE_TYPE (GannConvFilter, gann_conv_filter, G_TYPE_OBJECT);

enum
{
    PROP_0,
    PROP_WIDTH,
    PROP_HEIGHT,
    PROP_DEPTH,
    PROP_SIZE,
    N_PROPS,
};

static GParamSpec *props[N_PROPS];

static void set_property (GObject *gobj, guint propid,
                          const GValue *value, GParamSpec *spec);
static void get_property (GObject *gobj, guint propid,
                          GValue *value, GParamSpec *spec);
static void constructed (GObject *gobj);
static void finalize (GObject *gobj);

static void
gann_conv_filter_init (GannConvFilter *self)
{
}

static void
gann_conv_filter_class_init (GannConvFilterClass *cls)
{
    GObjectClass *gcls = G_OBJECT_CLASS (cls);

    gcls->set_property = set_property;
    gcls->get_property = get_property;
    gcls->constructed = constructed;
    gcls->finalize = finalize;

    props[PROP_WIDTH] =
        g_param_spec_int ("width",
                          "Width",
                          "Filter's width",
                          0, G_MAXINT32, 1,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS);

    props[PROP_HEIGHT] =
        g_param_spec_int ("height",
                          "Height",
                          "Filter's height",
                          0, G_MAXINT32, 1,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS);

    props[PROP_DEPTH] =
        g_param_spec_int ("depth",
                          "Depth",
                          "Filter's depth",
                          0, G_MAXINT32, 1,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS);

    props[PROP_SIZE] =
        g_param_spec_int ("size",
                          "Size",
                          "Filter's total size",
                          0, G_MAXINT32, 1,
                          G_PARAM_READABLE |
                          G_PARAM_STATIC_STRINGS);


    g_object_class_install_properties (gcls, N_PROPS, props);
}

static void
set_property (GObject *gobj, guint propid,
              const GValue *value, GParamSpec *spec)
{
    GannConvFilter *self = GANN_CONV_FILTER (gobj);

    switch (propid) {
    case PROP_WIDTH:
        self->width = g_value_get_int (value);
        break;

    case PROP_HEIGHT:
        self->height = g_value_get_int (value);
        break;

    case PROP_DEPTH:
        self->depth = g_value_get_int (value);
        break;

    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (gobj, propid, spec);
    }
}

static void
get_property (GObject *gobj, guint propid,
              GValue *value, GParamSpec *spec)
{
    GannConvFilter *self = GANN_CONV_FILTER (gobj);

    switch (propid) {
    case PROP_WIDTH:
        g_value_set_int (value, self->width);
        break;

    case PROP_HEIGHT:
        g_value_set_int (value, self->height);
        break;

    case PROP_DEPTH:
        g_value_set_int (value, self->depth);
        break;

    case PROP_SIZE:
        g_value_set_int (value, self->size);
        break;

    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (gobj, propid, spec);
    }
}

static void
constructed (GObject *gobj)
{
    GannConvFilter *self = GANN_CONV_FILTER (gobj);

    self->size = self->width * self->height * self->depth;
    self->value_v = g_new (gfloat, self->size);

    g_object_notify_by_pspec (gobj, props[PROP_SIZE]);

    G_OBJECT_CLASS (gann_conv_filter_parent_class)->constructed (gobj);
}

static void
finalize (GObject *gobj)
{
    GannConvFilter *self = GANN_CONV_FILTER (gobj);

    g_clear_pointer (&self->value_v, g_free);

    G_OBJECT_CLASS (gann_conv_filter_parent_class)->finalize (gobj);
}

GannConvFilter *
gann_conv_filter_new (gint width,
                      gint height,
                      gint depth)
{
    return g_object_new (GANN_TYPE_CONV_FILTER,
                         "width", width,
                         "height", height,
                         "depth", depth,
                         NULL);
}

GannConvFilter *
gann_conv_filter_write_row (GannConvFilter *self,
                            gint layer,
                            gint row,
                            const gfloat *value_v,
                            gsize value_c)
{
    gint i;

    g_return_val_if_fail (value_c == self->width, self);
    g_return_val_if_fail (layer < self->depth, self);
    g_return_val_if_fail (row < self->height, self);

    if (layer < 0) {
        for (i = 0; i < self->depth; i++) {
            gann_conv_filter_write_row (self, i, row, value_v, value_c);
        }
    }
    else if (row < 0) {
        for (i = 0; i < self->height; i++) {
            gann_conv_filter_write_row (self, layer, i, value_v, value_c);
        }
    } else {
        memcpy (self->value_v + row * self->width * self->depth,
                value_v, value_c * sizeof (gfloat));
    }

    return self;
}

GannConvFilter *
gann_conv_filter_write_column (GannConvFilter *self,
                               gint layer,
                               gint column,
                               const gfloat *value_v,
                               gsize value_c)
{
    gint row;

    g_return_val_if_fail (value_c == self->height, self);
    g_return_val_if_fail (layer < self->depth, self);
    g_return_val_if_fail (column < self->width, self);

    for (row = 0; row < self->width; row++) {
        gann_conv_filter_write_value (self, layer,
                                      column, row,
                                      value_v[row]);
    }

    return self;
}

GannConvFilter *
gann_conv_filter_write_value (GannConvFilter *self,
                              gint layer,
                              gint column,
                              gint row,
                              gfloat value)
{
    gint index;

    g_return_val_if_fail (layer < self->depth, self);
    g_return_val_if_fail (column < self->width, self);
    g_return_val_if_fail (row < self->height, self);

    index = row * self->width * self->depth
          + column * self->depth
          + layer;

    self->value_v[index] = value;

    return self;
}

gint
gann_conv_filter_get_width (GannConvFilter *self)
{
    return self->width;
}

gint
gann_conv_filter_get_height (GannConvFilter *self)
{
    return self->height;
}

gint
gann_conv_filter_get_depth (GannConvFilter *self)
{
    return self->depth;
}

gint
gann_conv_filter_get_size (GannConvFilter *self)
{
    return self->size;
}
