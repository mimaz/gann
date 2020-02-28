#pragma once

#include "gann-layer.h"

G_BEGIN_DECLS

#define GANN_TYPE_FULLY_LAYER (gann_fully_layer_get_type ())

G_DECLARE_FINAL_TYPE (GannFullyLayer, gann_fully_layer,
                      GANN, FULLY_LAYER, GannLayer);

G_END_DECLS
