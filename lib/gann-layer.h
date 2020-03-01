#pragma once

#include "gann-activation.h"

G_BEGIN_DECLS

struct layer;
typedef struct _GannNetwork GannNetwork;

#define GANN_TYPE_LAYER (gann_layer_get_type ())

G_DECLARE_DERIVABLE_TYPE (GannLayer, gann_layer,
                          GANN, LAYER, GObject);

struct _GannLayerClass
{
    GObjectClass parent_class;
};

GannLayer *gann_layer_new_input (GannNetwork *network,
                                 gint width,
                                 gint height,
                                 gint depth);
GannLayer *gann_layer_new_output (GannNetwork *network);
GannLayer *gann_layer_new_fully (GannNetwork *network,
                                 gint width,
                                 gint height,
                                 gint depth,
                                 GannActivation activation);

const gfloat *gann_layer_get_data (GannLayer *self,
                                   gsize *size);
struct layer *gann_layer_get_core (GannLayer *self);

GannNetwork *gann_layer_get_network (GannLayer *self);
gint gann_layer_get_width (GannLayer *self);
gint gann_layer_get_height (GannLayer *self);
gint gann_layer_get_depth (GannLayer *self);
GannActivation gann_layer_get_activation (GannLayer *self);

G_END_DECLS
