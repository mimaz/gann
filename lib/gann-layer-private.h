#pragma once

#include "core/core.h"

G_BEGIN_DECLS

struct layer;

typedef struct _GannLayer GannLayer;

void gann_layer_set_core (GannLayer *self,
                          struct layer *core);
enum activation_type gann_layer_get_core_activation (GannLayer *self);

G_END_DECLS
