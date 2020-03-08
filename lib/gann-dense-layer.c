#include "gann-dense-layer.h"

#include "gann-network.h"
#include "gann-layer-private.h"

#include "core/core.h"

struct _GannDenseLayer
{
    GannLayer parent_instance;
};

G_DEFINE_TYPE (GannDenseLayer, gann_dense_layer, GANN_TYPE_LAYER);

static void constructed (GObject *gobj);

static void
gann_dense_layer_init (GannDenseLayer *self)
{
}

static void
gann_dense_layer_class_init (GannDenseLayerClass *cls)
{
    GObjectClass *gcls = G_OBJECT_CLASS (cls);

    gcls->constructed = constructed;
}

static void
constructed (GObject *gobj)
{
    GannLayer *layer;
    GannNetwork *network;
    struct layer *core;

    layer = GANN_LAYER (gobj);
    network = gann_layer_get_network (layer);

    core = layer_make_full (gann_network_get_core (network),
                            gann_layer_get_core_activation (layer),
                            gann_layer_get_width (layer),
                            gann_layer_get_height (layer),
                            gann_layer_get_depth (layer));
    gann_layer_set_core (layer, core);

    G_OBJECT_CLASS (gann_dense_layer_parent_class)->constructed (gobj);
}
