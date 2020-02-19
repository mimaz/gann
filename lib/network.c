#include "network.h"
#include "layer.h"

struct network *
network_make_empty ()
{
  struct network *net;

  net = g_new (struct network, 1);
  net->layers = g_ptr_array_new ();

  return net;
}

struct layer *
network_layer (struct network *net, int index)
{
  int count;
  count = network_layer_count (net);
  if (index < 0) {
    index += count;
    g_assert (index >= 0);
    return network_layer (net, count + index);
  }
  g_assert (index < count);
  return g_ptr_array_index (net->layers, index);
}

struct layer *
network_layer_last (struct network *net)
{
  return network_layer (net, -1);
}

int
network_layer_count (struct network *net)
{
  return net->layers->len;
}

void
network_push_layer (struct network *net, struct layer *lay)
{
  g_ptr_array_insert (net->layers, -1, lay);
}

void
network_set_data (struct network *net, float *values, int size)
{
  net->data = values;
  net->data_size = size;
}

void
network_forward (struct network *net)
{
  int i, count;
  struct layer *lay;

  count = network_layer_count (net);

  for (i = 0; i < count; i++) {
    lay = network_layer (net, i);
    net->input = net->data;
    net->input_size = net->data_size;
    layer_forward (lay);
  }
}

void
network_backward (struct network *net)
{
}
