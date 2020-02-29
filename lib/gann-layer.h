#pragma once

#include "gann-activation.h"

G_BEGIN_DECLS

typedef struct _GannNetwork GannNetwork;

#define GANN_TYPE_LAYER (gann_layer_get_type ())

G_DECLARE_DERIVABLE_TYPE (GannLayer, gann_layer,
                          GANN, LAYER, GObject);

struct _GannLayerClass
{
    GObjectClass parent_class;
};

const gfloat *gann_layer_get_data (GannLayer *self,
                                   gsize *size);

GannNetwork *gann_layer_get_network (GannLayer *self);
gint gann_layer_get_width (GannLayer *self);
gint gann_layer_get_height (GannLayer *self);
gint gann_layer_get_depth (GannLayer *self);
GannActivation gann_layer_get_activation (GannLayer *self);

G_END_DECLS
