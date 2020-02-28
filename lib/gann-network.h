#pragma once

#include "gann-activation.h"

G_BEGIN_DECLS

typedef struct _GannLayer GannLayer;
typedef struct _GannInputLayer GannInputLayer;
typedef struct _GannFullyLayer GannFullyLayer;

#define GANN_TYPE_NETWORK (gann_network_get_type ())

G_DECLARE_DERIVABLE_TYPE (GannNetwork, gann_network,
                          GANN, NETWORK, GObject);

struct _GannNetworkClass
{
    GObjectClass parent_class;
};

GannNetwork *gann_network_new ();
GannInputLayer *gann_network_create_input (GannNetwork *self,
                                           gint width,
                                           gint height,
                                           gint depth);
GannFullyLayer *gann_network_create_fully (GannNetwork *self,
                                           gint width,
                                           gint height,
                                           gint depth,
                                           GannActivation activation);

GannInputLayer *gann_network_get_input_layer (GannNetwork *self);

void gann_network_set_rate (GannNetwork *self,
                            gfloat rate);
gfloat gann_network_get_rate (GannNetwork *self);

void gann_network_set_momentum (GannNetwork *self,
                                gfloat momentum);
gfloat gann_network_get_momentum (GannNetwork *self);

void gann_network_set_decay (GannNetwork *self,
                             gfloat decay);
gfloat gann_network_get_decay (GannNetwork *self);

G_END_DECLS
