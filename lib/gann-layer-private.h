#pragma once

#include "core/core.h"

G_BEGIN_DECLS

struct layer;

typedef struct _GannLayer GannLayer;

void gann_layer_set_core (GannLayer *self,
                          struct layer *core);

G_END_DECLS
