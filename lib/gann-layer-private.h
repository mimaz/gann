#pragma once

#include <glib.h>

G_BEGIN_DECLS

typedef struct _GannLayer GannLayer;
typedef struct _GannNetwork GannNetwork;
typedef struct _GannLayerPrivate GannLayerPrivate;

struct _GannLayerPrivate {
    GannNetwork *gnet;

    gint width;
    gint height;
    gint depth;

    struct layer *l;
};

GannLayerPrivate *gann_layer_get_private (gpointer self);
GannLayer *gann_layer_new_input (GannNetwork *network,
                                 gint width,
                                 gint height,
                                 gint depth);
GannLayer *gann_layer_new_fully (GannNetwork *network,
                                 gint width,
                                 gint height,
                                 gint depth);

G_END_DECLS
