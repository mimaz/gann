#pragma once

#include "gann-layer.h"

G_BEGIN_DECLS

#define GANN_TYPE_OUTPUT_LAYER (gann_output_layer_get_type ())

G_DECLARE_FINAL_TYPE (GannOutputLayer, gann_output_layer,
                      GANN, OUTPUT_LAYER, GannLayer);

void gann_output_layer_set_truth (GannOutputLayer *self,
                                  const gfloat *data,
                                  gsize datasize);
void gann_output_layer_set_truth_ints (GannOutputLayer *self,
                                       gint first, ...);

G_END_DECLS
