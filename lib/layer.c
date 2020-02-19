#include "layer.h"

void
layer_forward (struct layer *lay)
{
  lay->forward (lay);
}

void
layer_backward (struct layer *lay)
{
  lay->backward (lay);
}
