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
    GannLayerPrivate *p = gann_layer_get_private (gobj);

    p->l = layer_make_full (gann_network_get_core (p->gnet),
                            ACTIVATION_LEAKY,
                            p->width,
                            p->height,
                            p->depth);

    G_OBJECT_CLASS (gann_fully_layer_parent_class)->constructed (gobj);
}
