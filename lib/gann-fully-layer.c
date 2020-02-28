#include "gann-fully-layer.h"

#include "gann-layer-private.h"
#include "gann-network-private.h"

#include "core/layer.h"

struct _GannFullyLayer
{
    GannLayer parent_instance;
};

G_DEFINE_TYPE (GannFullyLayer, gann_fully_layer, GANN_TYPE_LAYER);

static void constructed (GObject *gobj);

static void
gann_fully_layer_init (GannFullyLayer *self)
{
}

static void
gann_fully_layer_class_init (GannFullyLayerClass *cls)
{
    GObjectClass *gcls = G_OBJECT_CLASS (cls);

    gcls->constructed = constructed;
}

static void
constructed (GObject *gobj)
{
    GannLayer *layer;
    GannNetwork *network;
    enum activation_type activation;
    struct layer *core;

    layer = GANN_LAYER (gobj);
    network = gann_layer_get_network (layer);

    switch (gann_layer_get_activation (layer)) {
    case GANN_ACTIVATION_LINEAR:
        activation = ACTIVATION_LINEAR;
        break;

    case GANN_ACTIVATION_RELU:
        activation = ACTIVATION_RELU;
        break;

    case GANN_ACTIVATION_SIGMOID:
        activation = ACTIVATION_SIGMOID;
        break;

    default:
        g_error ("invalid activation value");
    }

    core = layer_make_full (gann_network_get_core (network),
                            activation,
                            gann_layer_get_width (layer),
                            gann_layer_get_height (layer),
                            gann_layer_get_depth (layer));
    gann_layer_set_core (layer, core);

    G_OBJECT_CLASS (gann_fully_layer_parent_class)->constructed (gobj);
}
