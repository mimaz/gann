#pragma once

#include "gann-activation.h"

G_BEGIN_DECLS

struct network;
typedef struct _GannContext GannContext;
typedef struct _GannLayer GannLayer;
typedef struct _GannInputLayer GannInputLayer;
typedef struct _GannOutputLayer GannOutputLayer;
typedef struct _GannFullyLayer GannFullyLayer;

#define GANN_TYPE_NETWORK (gann_network_get_type ())

G_DECLARE_DERIVABLE_TYPE (GannNetwork, gann_network,
                          GANN, NETWORK, GObject);

struct _GannNetworkClass
{
    GObjectClass parent_class;
};

GannNetwork *gann_network_new (GannContext *context);
GannNetwork *gann_network_new_full (GannContext *context,
                                    gfloat rate,
                                    gfloat momentum,
                                    gfloat decay);
GannInputLayer *gann_network_create_input (GannNetwork *self,
                                           gint width,
                                           gint height,
                                           gint depth);
GannOutputLayer *gann_network_create_output (GannNetwork *self);
GannFullyLayer *gann_network_create_fully (GannNetwork *self,
                                           gint width,
                                           gint height,
                                           gint depth,
                                           GannActivation activation);
void gann_network_forward (GannNetwork *self);
void gann_network_backward (GannNetwork *self);
GannLayer *gann_network_get_layer (GannNetwork *self,
                                   gint index);
struct network *gann_network_get_core (GannNetwork *self);
void gann_network_forward (GannNetwork *self);
GannContext *gann_network_get_context (GannNetwork *self);
void gann_network_set_rate (GannNetwork *self,
                            gfloat rate);
gfloat gann_network_get_rate (GannNetwork *self);
void gann_network_set_momentum (GannNetwork *self,
                                gfloat momentum);
gfloat gann_network_get_momentum (GannNetwork *self);
void gann_network_set_decay (GannNetwork *self,
                             gfloat decay);
gfloat gann_network_get_decay (GannNetwork *self);
gint gann_network_get_layer_count (GannNetwork *self);
gfloat gann_network_get_loss (GannNetwork *self);
gfloat gann_network_get_average_loss (GannNetwork *self);

G_END_DECLS
