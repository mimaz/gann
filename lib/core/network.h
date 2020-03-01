#pragma once

#include <glib.h>

struct layer;

struct network
{
    GPtrArray *layers;
    float loss;
    float rate;
    float momentum;
    float decay;
};

struct network *network_make_empty ();
void network_free (struct network *net);
struct layer *network_layer (struct network *net, int index);
struct layer *network_layer_last (struct network *net);
int network_layer_count (struct network *net);
void network_push_layer (struct network *net, struct layer *lay);
void network_forward (struct network *net);
void network_backward (struct network *net);
void network_randomize (struct network *net);
