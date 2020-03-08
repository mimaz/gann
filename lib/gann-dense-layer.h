#pragma once

#include "gann-layer.h"

G_BEGIN_DECLS

#define GANN_TYPE_DENSE_LAYER (gann_dense_layer_get_type ())

G_DECLARE_FINAL_TYPE (GannDenseLayer, gann_dense_layer,
                      GANN, DENSE_LAYER, GannLayer);

G_END_DECLS
