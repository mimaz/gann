#pragma once

#include <glib.h>

G_BEGIN_DECLS

typedef struct _GannNetworkPrivate GannNetworkPrivate;

struct _GannNetworkPrivate
{
    struct network *net;
    GPtrArray *layer_arr;
};

struct network *gann_network_get_core (GannNetwork *self);
GannNetworkPrivate *gann_network_get_private (gpointer self);

G_END_DECLS
