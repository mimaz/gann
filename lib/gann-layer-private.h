#pragma once

#include "gann-activation.h"

G_BEGIN_DECLS

struct layer;

typedef struct _GannLayer GannLayer;
typedef struct _GannNetwork GannNetwork;

void gann_layer_set_core (GannLayer *self,
                          struct layer *core);
struct layer *gann_layer_get_core (GannLayer *self);
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

G_END_DECLS
