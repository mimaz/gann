#pragma once

#include "gann-layer.h"

G_BEGIN_DECLS

#define GANN_TYPE_INPUT_LAYER (gann_input_layer_get_type ())

G_DECLARE_FINAL_TYPE (GannInputLayer, gann_input_layer,
                      GANN, INPUT_LAYER, GannLayer);

void gann_input_layer_set_input (GannInputLayer *self,
                                 const gfloat *data,
                                 gsize datasize);
void gann_input_layer_set_input_floats (GannInputLayer *self,
                                        gfloat first, ...);

G_END_DECLS
