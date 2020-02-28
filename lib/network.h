#pragma once

#include <glib.h>

struct layer;

struct network
{
    GPtrArray *layers;
    float *input_v;
    float *data_v;
    float *delta_v;
    float *truth_v;
    int input_c;
    int data_c;
    int delta_c;
    int truth_c;
    float loss;
};

struct network *network_make_empty ();
struct layer *network_layer (struct network *net, int index);
struct layer *network_layer_last (struct network *net);
int network_layer_count (struct network *net);
void network_push_layer (struct network *net, struct layer *lay);
void network_set_data (struct network *net, float *values, int size);
void network_set_truth (struct network *net, float *values, int size);
void network_set_delta (struct network *net, float *value, int size);
void network_forward (struct network *net);
void network_backward (struct network *net);
void network_randomize (struct network *net);
