#include "layer.h"
#include "network.h"

static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);

struct layer *
layer_make_input (struct network *net,
    int width, int height, int depth)
{
  struct layer *base;
  int size;

  size = width * width * depth;

  base = g_new (struct layer, 1);
  base->net = net;
  base->type = LAYER_INPUT;
  base->weights = NULL;
  base->values = g_new (float, size);
  base->size = size;
  base->width = width;
  base->height = height;
  base->depth = depth;
  base->forward = forward;
  base->backward = backward;
  base->release = release;

  network_push_layer (net, base);

  return base;
}

static void
forward (struct layer *lay)
{
  g_assert (lay->net->input_size == lay->size);
  memcpy (lay->values, lay->net->input, sizeof (float) * lay->size);
  network_set_data (lay->net, lay->values, lay->size);
}

static void
backward (struct layer *lay)
{
}

static void
release (struct layer *lay)
{
  g_clear_pointer (&lay->values, g_free);
}
